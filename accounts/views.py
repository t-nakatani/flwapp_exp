from django.shortcuts import render

# Create your views here.
from django.shortcuts import redirect, render

from .models import User
from .forms import SignupForm, LoginForm
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.views.generic import CreateView

class SignUpView(CreateView):
    model = User
    form_class = SignupForm
    template_name = 'signup.html'

    def form_valid(self, form):
        user = form.save()
        login(self.request, user)
        return redirect('login')
