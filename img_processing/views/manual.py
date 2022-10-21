from django.contrib.auth.decorators import login_required
from django.http.response import HttpResponseForbidden, HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.db import transaction
from img_processing.models import ImageProcessing

SIZE_RATIO = 2.5
IMG_WIDTH = 400
IMG_HEIGHT = 265

@login_required
def arrangement(request, user_id):
    user = request.user
    if user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    if request.method == 'GET':
        context = {'path_img': f'/media/estimated/{user.next_img_id}/img.png',
                   'height': int(IMG_HEIGHT * 1.5),
                   'width': int(IMG_WIDTH * 1.5),
                   'user': user}

        return render(request, 'select_arrangement.html', context)

    if request.method == 'POST':
        predict = request.POST['predict']
        return redirect('manual_submit', user_id, predict)


@login_required
def submit(request, user_id, predict):
    """
    GET: 結果の提出前確認
    POST: ImageProcessing.predictとUser.next_img_idを更新
    """
    user = request.user
    if user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    if request.method == 'GET':
        context = {'path_img': f'/media/estimated/{user.next_img_id}/img.png',
                   'height': int(IMG_HEIGHT * 1.5),
                   'width': int(IMG_WIDTH * 1.5),
                   'predict': predict.replace('-', ''),
                   'filename': f'{predict}.svg'}
        return render(request, 'manual_submit.html', context)

    if request.method == 'POST':
        predict = request.POST['predict']
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
