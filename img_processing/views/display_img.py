from django.contrib.auth.decorators import login_required
from django.http.response import HttpResponseForbidden, HttpResponseRedirect
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from img_processing.models import ImageProcessing

from estimater import estimate
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
        processing = ImageProcessing.objects.create(user=user, img_id=user.next_img_id)
        processing.save()

        context = {
        'first_estimation': True,
        'path_img' : f'/media/estimated/{user.next_img_id}/img.png', 
        'path_img_corner' : f'/media/estimated/{user.next_img_id}/img_corner_.png', 
        'height' : IMG_HEIGHT, 
        'width' : IMG_WIDTH, 
        }
        print(context)
        return render(request, 'img_corner.html', context)
    if request.method == 'POST':
        clicked_coord = (request.POST.get('coord_list', None)).split(',')
        if clicked_coord[0] == '': # clickなしにPOSTが起こった場合．https://office54.net/python/django/display-message-framework
            messages.add_message(request, messages.ERROR, u"ERROR: 花弁の重なり位置を選択してから再推定ボタンを押下してください")
            return HttpResponseRedirect(request.path)
        clicked_coord = list(map(lambda x: int(int(x)*SIZE_RATIO/2), clicked_coord))
        clicked_coord = np.array(clicked_coord).reshape(-1, 2)
        
        estimate.re_infer_with_clicked(f'media/processing_data/user_{user.id}/img.png', clicked_coord)
        context = {
            'first_estimation': True,
            'path_img' : f'./processing_data/user_{user.id}/img.png', 
            'path_img_corner' : f'./processing_data/user_{user.id}/img_corner_.png', 
            'path_img_corner_old' : f'./processing_data/user_{user.id}/img_corner_old.png', 
            'height' : IMG_HEIGHT, 
            'width' : IMG_WIDTH, 
        }
        # return redirect(display_img_lr, user_id=user_id)
        return redirect('home')
