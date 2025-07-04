from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Cloud
from .serializers import CloudSerializer

class CloudAPIView(APIView):
    def get(self, request):
        data = Cloud.objects.all().order_by('-timestamp')
        serializer = CloudSerializer(data, many=True)
        return Response(serializer.data)