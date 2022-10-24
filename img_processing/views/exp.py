from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.http.response import HttpResponse, HttpResponseForbidden
from django.shortcuts import render, redirect
from img_processing.forms import QuestionnaireForm
from img_processing.models import Image
from django.views.generic import CreateView
from django.contrib import messages

from estimater import estimate
import shutil
import os

from img_processing.models import ImageProcessing


class register_img(CreateView):
    """実験の前準備として初回の花弁配置値の推定を完了して保存するためのクラス"""
    template_name = 'img_upload.html'
    model = Image
    fields = ('img_id', 'img')

    def get_success_url(self):
        img = Image.objects.order_by('id').last()
        os.mkdir(f'media/estimated/{img.img_id}')
        shutil.move(f'media/{img.img}', f'media/estimated/{img.img_id}/img.png')
        estimate.estimate_arr(f'media/estimated/{img.img_id}/img.png')
        return '/register_img'


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
        if user.use_system:
            if not os.path.exists(f'media/processing_data/user_{user.id}'):
                shutil.copytree(f'media/estimated/{user.next_img_id}', f'media/processing_data/user_{user.id}')
        processing, _ = ImageProcessing.objects.get_or_create(user=user, img_id=user.next_img_id)
        processing.save()

        if user.use_system:
            return redirect('img_corner', user_id)
        else:
            return redirect('select_arrangement', user_id)


@login_required
def questionnaire(request, user_id):
    """アンケートフォーム"""

    user = request.user
    if user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    if request.method == 'GET':
        form = QuestionnaireForm()
        context = {'form': form, '1to5': [1, 2, 3, 4, 5]}
        return render(request, 'questionnaire.html', context)

    if request.method == 'POST':
        form = QuestionnaireForm(request.POST)
        if form.is_valid():
            questionnaire = form.save(commit=False)
            questionnaire.usability = request.POST["radio_options"]
            questionnaire.user = user
            try:
                questionnaire.save()
            except Exception:
                messages.add_message(request, messages.ERROR, u"ERROR: 重複してアンケートが送信されました")
            return redirect('home')
