from img_processing.models import Image, Questionnaire
from django import forms


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = {'img_id', 'img'}


class QuestionnaireForm(forms.ModelForm):
    class Meta:
        model = Questionnaire
        fields = {'trouble', 'to_be_improved'}
        labels = {
            'trouble': '実験中困ったことがあれば教えてください',
            'to_be_improved': '改善点があれば教えてください',
        }

class BugReportForm(forms.Form):
    text = forms.CharField(label='バグの内容', max_length=100)
