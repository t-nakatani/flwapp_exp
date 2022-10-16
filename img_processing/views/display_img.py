from django.contrib.auth.decorators import login_required
from django.http.response import HttpResponse, HttpResponseForbidden, HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from estimater import estimate
from django.db import transaction
from img_processing.models import ImageProcessing
import numpy as np

SIZE_RATIO = 2.5
IMG_WIDTH = 400
IMG_HEIGHT = 265

@login_required
def corner(request, user_id):
    """花弁の重なり検出の結果を表示する機能"""

    user = request.user
    if user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    if request.method == 'GET':
        context = {'before_modification': True,
                   'path_img': f'/media/processing_data/user_{user.id}/img.png',
                   'path_img_corner': f'/media/processing_data/user_{user.id}/img_corner_.png',
                   'height': IMG_HEIGHT,
                   'width': IMG_WIDTH,
                   'user': user}
        return render(request, 'img_corner.html', context)

    if request.method == 'POST':
        clicked_coord = (request.POST.get('coord_list', None)).split(',')
        if clicked_coord[0] == '':  # clickなしにPOSTが起こった場合．https://office54.net/python/django/display-message-framework
            messages.add_message(request, messages.ERROR, u"ERROR: 花弁の重なり位置を選択してから再推定ボタンを押下してください")
            return HttpResponseRedirect(request.path)
        clicked_coord = list(map(lambda x: int(int(x) * SIZE_RATIO / 2), clicked_coord))
        clicked_coord = np.array(clicked_coord).reshape(-1, 2)

        estimate.re_infer_with_clicked(f'media/processing_data/user_{user.id}/img.png', clicked_coord)
        context = {'before_modification': False,
                   'path_img': f'/media/processing_data/user_{user.id}/img.png',
                   'path_img_corner': f'/media/processing_data/user_{user.id}/img_corner_.png',
                   'path_img_corner_old': f'/media/processing_data/user_{user.id}/img_corner_old.png',
                   'height': IMG_HEIGHT,
                   'width': IMG_WIDTH,
                   'user': user}
        return render(request, 'img_corner.html', context)


@login_required
def lr(request, user_id):
    """花弁の前後関係の推定結果を表示する機能"""

    user = request.user
    if user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    if request.method == 'GET':
        context = {'before_modification': True,
                   'path_img': f'/media/processing_data/user_{user.id}/img.png',
                   'path_img_lr': f'/media/processing_data/user_{user.id}/img_lr.png',
                   'height': IMG_HEIGHT,
                   'width': IMG_WIDTH,
                   'user': user}
        return render(request, 'img_lr.html', context)

    if request.method == 'POST':
        clicked_coord = (request.POST.get('coord_list', None)).split(',')
        if clicked_coord[0] == '':  # clickなしにPOSTが起こった場合．https://office54.net/python/django/display-message-framework
            messages.add_message(request, messages.ERROR, u"ERROR: 花弁の重なり位置を選択してから再推定ボタンを押下してください")
            return HttpResponseRedirect(request.path)

        clicked_coord = list(map(lambda x: int(int(x) * SIZE_RATIO / 2), clicked_coord))
        clicked_coord = np.array(clicked_coord).reshape(-1, 2)

        estimate.update_intersection_label(f'media/processing_data/user_{user.id}/img.png', clicked_coord)
        context = {'before_modification': False,
                   'path_img': f'/media/processing_data/user_{user.id}/img.png',
                   'path_img_lr': f'/media/processing_data/user_{user.id}/img_lr.png',
                   'path_img_lr_old': f'/media/processing_data/user_{user.id}/img_lr_old.png',
                   'height': IMG_HEIGHT,
                   'width': IMG_WIDTH,
                   'user': user}
        return render(request, 'img_lr.html', context=context)


@login_required
def submit(request, user_id):
    """
    GET: 結果の提出前確認
    POST: ImageProcessing.predictとUser.next_img_idを更新
    """
    user = request.user
    if user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    if request.method == 'GET':
        context = {'path_img_lr': f'/media/processing_data/user_{user.id}/img_lr.png',
                   'height': IMG_HEIGHT,
                   'width': IMG_WIDTH,
                   'user': request.user}
        return render(request, 'submit.html', context)

    if request.method == 'POST':
        predict = estimate.get_predict_from_csv(f'media/processing_data/user_{user.id}/df_n.csv')
        with transaction.atomic():
            processing = get_object_or_404(ImageProcessing, user=user, img_id=user.next_img_id)
            processing.predict = predict
            processing.save()
            user.next_img_id += 1
            user.save()

        if user.next_img_id == 20:
            messages.add_message(request, messages.SUCCESS, u"実験は終了です．お疲れ様でした．")
            return HttpResponseRedirect(request.path)
        return redirect('progress', user_id)
