from django.contrib.auth.decorators import login_required
from django.shortcuts import render


def home(request):
    """ホームページ"""
    return render(request, "home.html")

@login_required
def note(request):
    """実験注意書きページ"""
    return render(request, "note.html")
