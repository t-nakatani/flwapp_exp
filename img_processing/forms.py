from traceback import format_stack
from img_processing.models import Image
from django import forms


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = {'path', 'img'}
