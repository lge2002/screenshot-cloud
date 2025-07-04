from rest_framework import serializers
from .models import Cloud

class CloudSerializer(serializers.ModelSerializer):
    class Meta:
        model = Cloud
        fields = '__all__'
