from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.decorators import login_required
from django.http.response import HttpResponse, HttpResponseForbidden
from django.shortcuts import render, redirect, get_object_or_404
from img_processing.forms import QuestionnaireForm, BugReportForm
from img_processing.models import Image
from django.views.generic import CreateView
from django.contrib import messages
from django.db import transaction

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
        percentage_completed = f'{int(request.user.num_finished_img * 3.3) + 1}%'
        context = {'percentage_completed': percentage_completed,
                   'user': request.user}
        return render(request, 'progress.html', context)

    if request.method == 'POST':
        if user.use_system:
            if os.path.exists(f'media/processing_data/user_{user.id}'):
                shutil.rmtree(f'media/processing_data/user_{user.id}')
            shutil.copytree(f'media/estimated/{user.next_img_id}', f'media/processing_data/user_{user.id}')
        processing, _ = ImageProcessing.objects.get_or_create(user=user,
                                                              img_id=user.next_img_id,
                                                              use_system=user.use_system)

        processing.use_system = user.use_system
        processing.save()

        if not user.trial_finished:
            user.trial_finished = True
            user.save()

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
        labels = ['1: 使いにくかった', '2: やや使いにくかった', '3: 同程度であった', '4: やや使いやすかった', '5: 使いやすかった']
        context = {'form': form, 'labels': labels}
        return render(request, 'questionnaire.html', context)

    if request.method == 'POST':
        form = QuestionnaireForm(request.POST)
        if form.is_valid():
            questionnaire = form.save(commit=False)
            questionnaire.system_usability = request.POST["system_usability"][0]
            questionnaire.user = user
            try:
                questionnaire.save()
            except Exception:
                messages.add_message(request, messages.ERROR, u"ERROR: 重複してアンケートが送信されました")
            return redirect('home')


def bug_report(request, user_id):
    """
    GET: バグのレポート画面を提供
    POST: ImageProcessing.predictとUser.next_img_idを更新
    """
    user = request.user
    if user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    if request.method == 'GET':
        form = BugReportForm()
        context = {'form': form,
                   'user': request.user}
        return render(request, 'bug_report.html', context)

    if request.method == 'POST':
        with transaction.atomic():
            form = BugReportForm(request.POST)
            if form.is_valid():
                text = form.cleaned_data['text']
                with open('bug_report.txt', mode='a') as f:
                    f.write(f'user: {user.username}(id={user.id}), img_id: {user.next_img_id}, text: {text}\n')

            processing = get_object_or_404(ImageProcessing,
                                           user=user,
                                           img_id=user.next_img_id,
                                           use_system=user.use_system)
            processing.predict = ''
            processing.save()
            shutil.move(
                f'media/processing_data/user_{user.id}',
                f'media/processing_data_log/user_{user.id}_img_{user.next_img_id}'
            )
            user.set_next_img_id()
            user.save()

        if user.num_finished_img == 30:
            messages.add_message(request, messages.SUCCESS, u"実験は終了です．アンケートにご協力ください．")
            return redirect('questionnaire', user_id)
        return redirect('progress', user_id)

@login_required
def trial_env(request, user_id):
    """
    system利用環境の操作手順確認用
    """
    user = request.user
    if user.id != user_id:
        return HttpResponseForbidden('You cannot access this page')

    if request.method == 'GET':
        return render(request, 'trial_home.html')

    if request.method == 'POST':
        if user.trial_finished:
            user.trial_finished = False
            user.save()

        if 'system' in request.POST:
            if os.path.exists(f'media/processing_data/user_{user.id}'):
                shutil.rmtree(f'media/processing_data/user_{user.id}')
            shutil.copytree('media/trial/sample', f'media/processing_data/user_{user.id}')
            return redirect('img_corner', user_id)
        else:  # manual
            return redirect('select_arrangement', user_id)
