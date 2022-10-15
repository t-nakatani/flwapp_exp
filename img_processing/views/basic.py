from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http.response import HttpResponseForbidden
import shutil, os

def save_result():
    """推定結果を保存する"""
    pass


def update_data_dir(user):
    """完了した推定結果を保存し次の推定結果を作業ディレクトリにセットする"""
    save_result()
    shutil.rmtree(f'data_{user.id}')  # TODO: update
    shutil.copytree(os.path.join('estimated', str(user.next_img_id)), f'data_{user.id}')


def home(request):
    """ホームページ"""
    return render(request, 'home.html')


@login_required
def note(request):
    """実験注意書きページ"""
    return render(request, 'note.html', {'user_id': request.user.id})


@login_required
def progress(request, user_id):
    """
    進捗確認ページ
    裏でset_next_img()を呼び出す
    """
    if request.user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    percentage_completed = f'{request.user.next_img_id * 5}%'
    context = {
        'percentage_completed': percentage_completed, 
        'user': request.user
    }
    return render(request, 'progress.html', context)
