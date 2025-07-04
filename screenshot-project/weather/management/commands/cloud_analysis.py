from django.core.management.base import BaseCommand
from django.conf import settings
from weather.models import Cloud
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from datetime import datetime, timedelta, date
import time
import os
import json
from django.template.loader import render_to_string
import cv2
from xhtml2pdf import pisa

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.ops import unary_union

import requests
import pytz


class Command(BaseCommand):
    help = 'Automates screenshot capture from Windy.com, crops to ALL Tamil Nadu districts, masks with shapefile, and analyzes cloud levels.'

    BLUE_DOT_XPATH = '//*[@id="leaflet-map"]/div[1]/div[4]/div[2]'
    API_ENDPOINT_URL = "http://172.16.7.118:8003/api/tamilnadu/satellite/push.windy_radar_data.php"
    VISIBLE_LAYER = '/html/body/span[1]/div/div[1]/div[6]/div[1]/div[2]'

    def _link_callback(self, uri, rel):
        """
        Convert HTML URIs to absolute system paths so xhtml2pdf can access them.
        This is crucial for local images specified with 'file:///' protocol.
        """
        # Remove 'file:///' prefix and normalize path for xhtml2pdf
        if uri.startswith('file:///'):
            # os.path.normpath handles / and \ appropriately for the OS
            # but xhtml2pdf's internal parser still prefers forward slashes
            # so we ensure it's converted to forward slashes after normpath
            return os.path.normpath(uri[len('file:///'):]).replace(os.path.sep, '/')
        
        sUrl = settings.STATIC_URL
        sRoot = settings.STATIC_ROOT
        mUrl = settings.MEDIA_URL
        mRoot = settings.MEDIA_ROOT

        if uri.startswith(mUrl):
            path = os.path.join(mRoot, uri.replace(mUrl, ""))
        elif uri.startswith(sUrl):
            path = os.path.join(sRoot, uri.replace(sUrl, ""))
        else:
            return uri

        if os.path.exists(path):
            # Ensure path returned to xhtml2pdf also uses forward slashes
            return path.replace(os.path.sep, '/')
        else:
            self.stderr.write(self.style.WARNING(f"Warning: Linked file not found: {path} for URI: {uri}"))
            return uri

    def _generate_and_save_automation_pdf(self, results_data, current_time, base_folder,
                                          full_screenshot_path_abs, cropped_screenshot_path_abs,
                                          json_output_content):
        """
        Generates a PDF report for a single automation run using xhtml2pdf and saves it to a file.
        """
        pdf_filename = f"automation_report_{current_time.strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_output_path = os.path.join(base_folder, pdf_filename)

        # Normalize paths for context to ensure forward slashes and correct file:/// protocol
        # The .replace(os.path.sep, '/') is critical for Windows paths with xhtml2pdf
        full_screenshot_path_for_html = full_screenshot_path_abs.replace(os.path.sep, '/')
        cropped_screenshot_path_for_html = cropped_screenshot_path_abs.replace(os.path.sep, '/')

        # Prepare the image paths in results_data for the template as well
        processed_results_data_for_pdf = []
        for item in results_data:
            new_item = item.copy()
            processed_results_data_for_pdf.append(new_item)

        context = {
            'current_time': current_time,
            'current_run_results': processed_results_data_for_pdf,
            'full_screenshot_path_abs': f'file:///{full_screenshot_path_for_html}',
            'cropped_screenshot_path_abs': f'file:///{cropped_screenshot_path_for_html}',
            'json_output_content': json_output_content,
        }

        html_string = render_to_string('weather/automation_report_pdf.html', context)

        with open(pdf_output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                html_string,
                dest=pdf_file,
                link_callback=self._link_callback
            )

        if pisa_status.err:
            raise Exception(f"PDF generation error with xhtml2pdf: {pisa_status.err}")

    def _round_to_nearest_minutes(self, dt_object, minutes=15):
        """
        Rounds a datetime object to the nearest specified number of minutes.
        E.g., for 15 minutes:
        10:00:00 -> 10:00:00
        10:07:29 -> 10:00:00
        10:07:30 -> 10:15:00 (rounds up at half the interval, i.e., 7 minutes 30 seconds)
        10:22:29 -> 10:15:00
        10:22:30 -> 10:30:00
        """
        total_minutes = dt_object.hour * 60 + dt_object.minute + dt_object.second / 60.0
        rounded_total_minutes = round(total_minutes / minutes) * minutes
        diff_minutes = rounded_total_minutes - total_minutes
        rounded_dt = dt_object + timedelta(minutes=diff_minutes)
        rounded_dt = rounded_dt.replace(second=0, microsecond=0)
        
        return rounded_dt

    def handle(self, **kwargs):
        self.stdout.write(self.style.SUCCESS('Starting Windy.com cloud analysis automation for all Tamil Nadu districts...'))

        shapefile_path = "C:/Users/tamilarasans/Downloads/gadm41_IND_2.json/gadm41_IND_2.json"
        if not os.path.exists(shapefile_path):
            self.stderr.write(self.style.ERROR(f"Critical Error: Shapefile not found at {shapefile_path}. Exiting."))
            return

        try:
            gdf = gpd.read_file(shapefile_path)
            tamil_nadu_gdf = gdf[gdf['NAME_1'] == 'TamilNadu']
            if tamil_nadu_gdf.empty:
                self.stderr.write(self.style.ERROR("Error: 'TamilNadu' not found in shapefile under 'NAME_1'. Please check the shapefile content."))
                return

            tamil_nadu_gdf = tamil_nadu_gdf.to_crs("EPSG:4326")

            all_tn_districts = tamil_nadu_gdf['NAME_2'].unique().tolist()
            if not all_tn_districts:
                self.stderr.write(self.style.ERROR("Error: No districts found for 'TamilNadu' under 'NAME_2' in shapefile. Exiting."))
                return

            self.stdout.write(f"Found {len(all_tn_districts)} districts in Tamil Nadu: {', '.join(all_tn_districts)}")

        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Error loading or processing shapefile initially: {e}. Exiting."))
            return

        while True:
            self.stdout.write("\n" + "="*50)
            self.stdout.write("STARTING NEW 5-MINUTE CYCLE: Capturing fresh screenshot and performing initial analysis.")
            self.stdout.write("="*50 + "\n")

            current_raw_time = datetime.now()
            
            # This rounds the timestamp for the data itself to the nearest 15 minutes,
            # as previously defined in your rounding function.
            current_time = self._round_to_nearest_minutes(current_raw_time, minutes=15)
            
            self.stdout.write(f"Raw capture time: {current_raw_time.strftime('%Y-%m-%d %H:%M:%S.%f')}")
            self.stdout.write(self.style.SUCCESS(f"Rounded analysis time for data: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"))

            timestamp_str = current_time.strftime('%Y-%m-%d_%H-%M-%S')

            base_folder = os.path.join(settings.BASE_DIR, "images", timestamp_str)
            full_image_folder = os.path.join(base_folder, "full")
            cropped_image_folder = os.path.join(base_folder, "cropped")
            os.makedirs(full_image_folder, exist_ok=True)
            os.makedirs(cropped_image_folder, exist_ok=True)

            full_screenshot_path = os.path.join(full_image_folder, "windy_map_full.png")
            cropped_screenshot_path = os.path.join(cropped_image_folder, "tamil_nadu_cropped.png")

            # CROP_BOX remains the same for the region of interest
            CROP_BOX = (551, 170, 1065, 687) # Left, Upper, Right, Lower

            driver = None
            current_run_results = []

            try:
                chrome_options = webdriver.ChromeOptions()
                chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
                chrome_options.add_experimental_option('useAutomationExtension', False)
                chrome_options.add_argument("--window-size=1920,1080")
                # chrome_options.add_argument("--headless")
                # chrome_options.add_argument("--disable-gpu")
                # chrome_options.add_argument("--no-sandbox")

                driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

                # --- MODIFICATION: Change URL to focus on 'Clouds' layer ---
                # The 'clouds' layer is typically accessed via '/-clouds' path.
                # Adjust coordinates as needed for proper centering on Tamil Nadu.
                driver.get("https://www.windy.com/-Satellite-satellite?satellite,10.950,77.500,7,p:favs")
                self.stdout.write(f"Navigated to Windy.com with satellite layer active and offset coordinates.")

                wait = WebDriverWait(driver, 20)

                try:
                    self.stdout.write('Attempting to dismiss cookie consent...')
                    cookie_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'button.cc-dismiss, a[aria-label="dismiss cookie message"]')))
                    cookie_button.click()
                    self.stdout.write('Cookie consent dismissed.')
                    time.sleep(1)

                except Exception as e:
                    self.stdout.write(f"Could not find or dismiss cookie consent (might not be present): {e}. Continuing...")
                    pass

                self.stdout.write("Waiting for map to fully load (10 seconds)...")
                time.sleep(10)

                self.stdout.write("Attempting to hide the blue dot using JavaScript injection with confirmed XPath...")
                try:
                    visible = wait.until(EC.element_to_be_clickable((By.XPATH, self.VISIBLE_LAYER)))
                    visible.click()
                    self.stdout.write("Successfully clicked the button/layer switch.")
                    time.sleep(5)  
                except Exception as e:
                    self.stderr.write(self.style.WARNING(f"Could not click the button via XPath. Check the XPath or page load state. Error: {e}"))


                try:
                    dot_element = wait.until(EC.presence_of_element_located((By.XPATH, self.BLUE_DOT_XPATH)))
                    driver.execute_script("arguments[0].style.display = 'none';", dot_element)
                    self.stdout.write("SUCCESS (Attempted): Dot element's display set to 'none' via JavaScript using confirmed XPath.")
                    self.stdout.write("NOTE: This method is often ineffective for elements drawn on a canvas, the dot may still be visible in the screenshot.")
                    time.sleep(1)   
                except Exception as e:
                    self.stdout.write(f"FAILED to hide dot via JavaScript at XPath '{self.BLUE_DOT_XPATH}': {e}.")
                    self.stdout.write("The element might not be present by this XPath, or another issue occurred. Trying fallback interactive methods (ESC key only)...")

                    try:
                        self.stdout.write("Fallback: Trying to press ESC key.")
                        ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                        self.stdout.write("Pressed ESC key to dismiss dot.")
                        time.sleep(1)
                    except Exception as esc_e:
                        self.stdout.write(f"Fallback (ESC key) failed: {esc_e}. The dot might still be visible.")
                
                time.sleep(2)

                self.stdout.write(f"Taking full screenshot and saving to: {full_screenshot_path}")
                driver.save_screenshot(full_screenshot_path)
                self.stdout.write("Screenshot saved successfully.")

            except Exception as e:
                self.stderr.write(self.style.ERROR(f"An unexpected error occurred during browser automation: {e}"))
                self.stdout.write("Waiting 5 minutes before retry...\n")
                time.sleep(300)
                continue
            finally:
                if driver:
                    driver.quit()
                    self.stdout.write("Browser closed.")

            # --- Image processing and initial analysis for ALL districts (runs once per 5-min cycle) ---
            try:
                image = Image.open(full_screenshot_path).convert("RGB")
                if not (0 <= CROP_BOX[0] < CROP_BOX[2] <= image.width and
                                0 <= CROP_BOX[1] < CROP_BOX[3] <= image.height):
                    self.stderr.write(self.style.ERROR("CROP_BOX coordinates are out of bounds. Skipping all district analysis for this run."))
                    time.sleep(300)
                    continue

                cropped_image = image.crop(CROP_BOX)
                cropped_image.save(cropped_screenshot_path)
                self.stdout.write(f"Cropped Tamil Nadu image saved at: {cropped_screenshot_path}")

                img_pil = cropped_image.convert("RGB")
                img_np = np.array(img_pil)
                height, width, _ = img_np.shape

                final_min_lon = 74.80
                final_max_lon = 80.37
                final_min_lat = 7.98
                final_max_lat = 13.53

                transform = from_bounds(final_min_lon, final_min_lat, final_max_lon, final_max_lat, width, height)

                # --- MODIFICATION: Cloud Percentage Analysis Logic ---
                # Define a range of colors that typically represent clouds on satellite imagery.
                # These thresholds might need fine-tuning based on actual Windy.com cloud layer colors.
                # Clouds are generally white/gray/light blue. Non-cloud areas (land/sea) would be
                # greens, browns, deep blues.
                
                # A simple approach: consider pixels with high R, G, B values (closer to white) as clouds.
                # Or, pixels that are not close to typical land/sea colors.
                # Let's start with a simple threshold for brightness/whiteness.
                # We can refine this by looking at specific RGB ranges if necessary.



                def is_cloud_pixel(rgb_pixel):
                    r, g, b = rgb_pixel

                    # Perceptual grayscale brightness
                    brightness = 0.299 * r + 0.587 * g + 0.114 * b

                    # Step 1: Quick brightness check (fast filter)
                    if brightness < 160:
                        return False  # Too dark to be a cloud

                    # Step 2: Convert to HSV color space for better color analysis
                    hsv_pixel = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
                    h, s, v = hsv_pixel  # Hue, Saturation, Value

                    # Step 3: Define HSV-based cloud criteria
                    # Clouds are usually:
                    # - High Value (brightness)
                    # - Low to medium Saturation (not colorful)
                    # - Hue can vary, but not highly saturated

                    if v > 180 and s < 80:
                        return True

                    # Optional fallback: light bluish-gray
                    if r > 170 and g > 170 and b > 190:
                        return True

                    return False



                for district_name in all_tn_districts:
                    self.stdout.write(f"\nProcessing district: {district_name} for cloud percentage and DB save...")

                    district_masked_folder = os.path.join(base_folder, "masked_cropped", district_name.replace(" ", "_"))
                    os.makedirs(district_masked_folder, exist_ok=True)
                    masked_cropped_path = os.path.join(district_masked_folder, f"{timestamp_str}_{district_name.lower().replace(' ', '_')}_masked.png")
                    
                    current_district_gdf = tamil_nadu_gdf[tamil_nadu_gdf['NAME_2'] == district_name]

                    if current_district_gdf.empty:
                        self.stderr.write(self.style.WARNING(f"Warning: {district_name} not found in the filtered Tamil Nadu shapefile data. Skipping."))
                        continue

                    district_polygon = unary_union(current_district_gdf.geometry.to_list())

                    mask = rasterize(
                        [district_polygon],
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        all_touched=True,
                        dtype=np.uint8
                    )

                    transparent_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                    original_rgba_image = cropped_image.convert("RGBA")
                    original_rgba_np = np.array(original_rgba_image)

                    total_district_pixels = 0
                    cloudy_pixels = 0

                    for y in range(height):
                        for x in range(width):
                            if mask[y, x]: # If this pixel is part of the district
                                r, g, b, a = original_rgba_np[y, x]
                                if a > 0: # Ensure it's not a fully transparent pixel from the original crop edge
                                    total_district_pixels += 1
                                    transparent_image.putpixel((x, y), (r, g, b, 255)) # Re-add pixel to masked image
                                    if is_cloud_pixel((r, g, b)):
                                        cloudy_pixels += 1
                                    
                    transparent_image.save(masked_cropped_path)
                    self.stdout.write(f"Masked image of {district_name} saved at: {masked_cropped_path}")

                    cloud_percentage = 0.0
                    if total_district_pixels > 0:
                        cloud_percentage = (cloudy_pixels / total_district_pixels) * 100
                    
                    cloud_percentage_str = f"{cloud_percentage:.2f}%"
                    self.stdout.write(f"Cloud analysis for {district_name}: {cloud_percentage_str} cloud coverage.")

                    # --- START MODIFICATION FOR TIMEZONE LOCALIZATION ---
                    timestamp_for_db = current_time 
                    if settings.USE_TZ:
                        # Ensure the timestamp is localized to your project's timezone before saving
                        target_tz = pytz.timezone(settings.TIME_ZONE)
                        timestamp_for_db = target_tz.localize(timestamp_for_db)
                    # --- END MODIFICATION ---

                    try:
                        Cloud.objects.update_or_create(
                            city=district_name,
                            timestamp=timestamp_for_db, # This is now a timezone-aware datetime object
                            defaults={
                                "values": cloud_percentage_str, # Store percentage as string
                                "type": "Cloud Coverage" # Changed type to reflect new analysis
                            }
                        )
                        self.stdout.write(self.style.SUCCESS(f"Cloud analysis for {district_name} saved to database."))
                    except Exception as e:
                        self.stderr.write(self.style.ERROR(f"Error saving {district_name} to Django model: {e}"))

                    district_data_for_post_collection = { 
                        "city": district_name,
                        "values": cloud_percentage_str, # Use the percentage string
                        "type": "Cloud Coverage",
                        "timestamp": timestamp_for_db.strftime('%Y-%m-%d %H:%M:%S') # Use the localized timestamp for JSON
                    }
                    current_run_results.append(district_data_for_post_collection)
            
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error during initial image processing or shapefile handling for all districts: {e}"))
                self.stdout.write("Waiting 5 minutes before retry...\n")
                time.sleep(300)
                continue


            # --- Save the collected JSON data locally (once per 5-min cycle) ---
            json_filename = f"cloud_analysis_results_{timestamp_str}.json"
            json_output_path = os.path.join(base_folder, json_filename)
            json_output_content = json.dumps(current_run_results, indent=4) 
            try:
                with open(json_output_path, "w") as json_file:
                    json_file.write(json_output_content)
                self.stdout.write(self.style.SUCCESS(f"All initial analysis results for this 5-min cycle saved to JSON at: {json_output_path}"))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error saving full cycle JSON file: {e}"))

            # --- Generate and Save PDF Report for this cycle ---
            self.stdout.write("Generating PDF report for this run...")
            try:
                full_screenshot_path_abs = os.path.abspath(full_screenshot_path)
                cropped_screenshot_path_abs = os.path.abspath(cropped_screenshot_path)

                self._generate_and_save_automation_pdf(
                    current_run_results,
                    current_time, 
                    base_folder,
                    full_screenshot_path_abs,
                    cropped_screenshot_path_abs,
                    json_output_content
                )
                pdf_output_filename_for_message = f"automation_report_{current_time.strftime('%Y%m%d_%H%M%S')}.pdf"
                pdf_full_path_for_message = os.path.join(base_folder, pdf_output_filename_for_message)
                self.stdout.write(self.style.SUCCESS(f"PDF report generated and saved successfully to: {pdf_full_path_for_message}"))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error generating PDF report for this run: {e}"))

            # --- URL PUSHING (Now runs only 1 time per main cycle) ---
            num_post_attempts = 1

            for i in range(num_post_attempts):
                self.stdout.write(f"\n--- URL PUSHING ATTEMPT {i + 1} of {num_post_attempts} (using data from this cycle) ---")
                
                if self.API_ENDPOINT_URL and current_run_results:
                    self.stdout.write(f"Attempting to send ALL analysis data to {self.API_ENDPOINT_URL} via POST (Attempt {i+1})...")
                    
                    headers = {
                        'Content-Type': 'application/json',
                    }

                    try:
                        self.stdout.write(f"Sending JSON payload: {json.dumps(current_run_results, indent=4)}")
                        response = requests.post(self.API_ENDPOINT_URL, json=current_run_results, headers=headers, timeout=30)
                        response.raise_for_status()

                        self.stdout.write(self.style.SUCCESS(f"Data successfully POSTed to {self.API_ENDPOINT_URL} (Attempt {i+1})."))
                        self.stdout.write(f"API Response Status Code: {response.status_code}")
                        try:
                            self.stdout.write(f"API Response JSON: {response.json()}")
                        except json.JSONDecodeError:
                            self.stdout.write(f"API Response Text: {response.text}")
                    except requests.exceptions.HTTPError as http_err:
                        self.stderr.write(self.style.ERROR(f"HTTP error during POST request (Attempt {i+1}): {http_err}"))
                        if http_err.response:
                            self.stderr.write(self.style.ERROR(f"Response from API (Attempt {i+1}): {http_err.response.text}"))
                    except requests.exceptions.ConnectionError as conn_err:
                        self.stderr.write(self.style.ERROR(f"Connection error during POST request (Attempt {i+1}, Is the server at {self.API_ENDPOINT_URL} reachable and port open?): {conn_err}"))
                    except requests.exceptions.Timeout as timeout_err:
                        self.stderr.write(self.style.ERROR(f"Timeout error during POST request (Attempt {i+1}, API took too long to respond): {timeout_err}"))
                    except requests.exceptions.RequestException as req_err:
                        self.stderr.write(self.style.ERROR(f"An unexpected error occurred during POST request (Attempt {i+1}): {req_err}"))
                else:
                    if not self.API_ENDPOINT_URL:
                        self.stdout.write(self.style.WARNING(f"API_ENDPOINT_URL is not set. Skipping POST request (Attempt {i+1})."))
                    if not current_run_results:
                        self.stdout.write(self.style.WARNING(f"No analysis results to send. Skipping POST request (Attempt {i+1})."))
                
            
            self.stdout.write("\nFinished all URL pushing cycles for this data set.")
            self.stdout.write("Waiting 5 minutes before starting a new full run (fresh screenshot and analysis)....\n")
            time.sleep(300)