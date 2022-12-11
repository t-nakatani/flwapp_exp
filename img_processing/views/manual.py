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
        context = {'path_img': f'/media/estimated/{user.next_img_id}/img_bb.png',
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
        context = {'path_img': f'/media/estimated/{user.next_img_id}/img_bb.png',
                   'height': int(IMG_HEIGHT * 1.5),
                   'width': int(IMG_WIDTH * 1.5),
                   'predict': predict.replace('-', ''),
                   'filename': f'{predict}.svg',
                   'trial': not request.user.trial_finished}
        return render(request, 'manual_submit.html', context)

    if request.method == 'POST':
        if not user.trial_finished:
            user.trial_finished = True
            user.save()
            return redirect('trial', user_id)

        predict = request.POST['predict']
        with transaction.atomic():
            processing = get_object_or_404(ImageProcessing, user=user, img_id=user.next_img_id)
            processing.predict = predict
            processing.save()
            user.set_next_img_id()
            user.save()

        if user.num_finished_img == 30:
            messages.add_message(request, messages.SUCCESS, u"実験は終了です．アンケートにご協力ください．")
            return redirect('questionnaire', user_id)
        return redirect('progress', user_id)
