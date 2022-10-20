from django.contrib.auth.decorators import login_required
from django.shortcuts import render
import shutil
import os

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
    return render(request, 'home.html', {'user_id': request.user.id})


def note(request):
    """実験注意書きページ"""
    return render(request, 'note.html')


def agree(request):
    """実験同意ページ"""
    return render(request, 'agree.html')
