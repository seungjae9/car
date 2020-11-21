from django.db import models
from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill

# Create your models here.

class Image(models.Model):
    image = ProcessedImageField(
        processors= [ResizeToFill(300,300)],
        format= 'JPEG',
        options= {'quality': 90},
        upload_to= 'media',
    )