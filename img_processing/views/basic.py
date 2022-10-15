from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http.response import HttpResponseForbidden

def home(request):
    """ホームページ"""
    return render(request, 'home.html')


@login_required
def note(request):
    """実験注意書きページ"""
    return render(request, 'note.html', {'user_id': request.user.id})


@login_required
def progress(request, user_id):
    """進捗確認ページ"""
    if request.user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    percentage_completed = f'{request.user.next_img_id * 5}%'
    context = {
        'percentage_completed': percentage_completed, 
        'next_img_id': request.user.next_img_id
    }
    return render(request, 'progress.html', context)
