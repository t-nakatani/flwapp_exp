from django.contrib.auth.decorators import login_required
from django.http.response import HttpResponse, HttpResponseForbidden
from django.shortcuts import render, redirect
import shutil

from img_processing.models import ImageProcessing


@login_required
def register_img(request):
    """実験の前準備として初回の花弁配置値の推定を完了して保存するための機能"""

    if request.method == 'POST':
        pass

    else:
        if request.user.id == 1:
            return HttpResponse('user==nakatani OK')


@login_required
def progress(request, user_id):
    """
    GET: 進捗確認
    POST: ImageProcessingレコードの作成
    """
    user = request.user
    if user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    if request.method == 'GET':
        percentage_completed = f'{request.user.next_img_id * 5}%'
        context = {'percentage_completed': percentage_completed,
                   'user': request.user}
        return render(request, 'progress.html', context)

    if request.method == 'POST':
        shutil.copytree(f'media/estimated/{user.next_img_id}', f'media/processing_data/user_{user.id}')
        processing, _ = ImageProcessing.objects.get_or_create(user=user, img_id=user.next_img_id)
        processing.save()

        return redirect('img_corner', user_id)
