from django.contrib.auth.decorators import login_required
from django.http.response import HttpResponse, HttpResponseForbidden, HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from estimater import estimate
from django.db import transaction
from img_processing.models import ImageProcessing
import numpy as np
import shutil
import string
import random
import os

SIZE_RATIO = 2.5
IMG_WIDTH = 400
IMG_HEIGHT = 265

def cache_busting(path_img, n=4, start_with_slash=True):
    """ブラウザのキャッシュによって画像が更新されないことを防ぐためのcache busting"""
    dir_, fname = os.path.split(path_img)
    if start_with_slash:
        dir_ = dir_[1:]
        path_img = path_img[1:]
    fname, ftype = fname.split('.')
    randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    rand_string = ''.join(randlst)
    new_path = f'{dir_}/{fname}_{rand_string}.{ftype}'
    shutil.copy(path_img, new_path)
    if start_with_slash:
        new_path = '/' + new_path
    return new_path

@login_required
def corner(request, user_id):
    """花弁の重なり検出の結果を表示する機能"""

    user = request.user
    if user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    if request.method == 'GET':
        context = {'before_modification': True,
                   'path_img': cache_busting(f'/media/processing_data/user_{user.id}/img.png'),
                   'path_img_corner': cache_busting(f'/media/processing_data/user_{user.id}/img_corner.png'),
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
                   'path_img': cache_busting(f'/media/processing_data/user_{user.id}/img.png'),
                   'path_img_corner': cache_busting(f'/media/processing_data/user_{user.id}/img_corner.png'),
                   'path_img_corner_old': cache_busting(f'/media/processing_data/user_{user.id}/img_corner_old.png'),
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
                   'path_img': cache_busting(f'/media/processing_data/user_{user.id}/img.png'),
                   'path_img_lr': cache_busting(f'/media/processing_data/user_{user.id}/img_lr.png'),
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
                   'path_img': cache_busting(f'/media/processing_data/user_{user.id}/img.png'),
                   'path_img_lr': cache_busting(f'/media/processing_data/user_{user.id}/img_lr.png'),
                   'path_img_lr_old': cache_busting(f'/media/processing_data/user_{user.id}/img_lr_old.png'),
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
        context = {'path_img_lr': cache_busting(f'/media/processing_data/user_{user.id}/img_lr.png'),
                   'height': IMG_HEIGHT,
                   'width': IMG_WIDTH,
                   'user': request.user}
        return render(request, 'use_system_submit.html', context)

    if request.method == 'POST':
        predict = estimate.get_predict_from_csv(f'media/processing_data/user_{user.id}/df_n.csv')
        with transaction.atomic():
            processing = get_object_or_404(ImageProcessing, user=user, img_id=user.next_img_id)
            processing.predict = predict
            processing.save()
            shutil.move(
                f'media/processing_data/user_{user.id}',
                f'media/processing_data_log/user_{user.id}_img_{user.next_img_id}'
            )
            user.next_img_id += 1
            user.save()

        if user.next_img_id == 20:
            messages.add_message(request, messages.SUCCESS, u"実験は終了です．アンケートにご協力ください")
            return redirect('questionnare', user_id)
        return redirect('progress', user_id)
