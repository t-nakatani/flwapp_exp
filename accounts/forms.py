from .models import User
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

class SignupForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = User
        fields = ['username']

class LoginForm(AuthenticationForm):
    class Meta:
        model = User
        fields = ['username']