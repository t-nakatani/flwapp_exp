from django.shortcuts import redirect, render

from .models import User
from .forms import SignupForm, LoginForm
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.views.generic import CreateView

# Create your views here.

class SignUpView(CreateView):
    model = User
    form_class = SignupForm
    template_name = 'signup.html'

    def form_valid(self, form):
        """
        ユーザー作成時のform_valid
        use_systemは2の剰余で決定
        """
        user = form.save()
        user.use_system = user.id % 2
        user.set_next_img_id(finished=False)
        user.save()
        login(self.request, user)
        return redirect('login')
