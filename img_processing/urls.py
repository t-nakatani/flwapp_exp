from django.urls import path
from img_processing.views import basic, exp, display_img


urlpatterns = [
    path('register_img/', exp.register_img, name='register_img'),

    path('', basic.home, name='home'),
    path('note/', basic.note, name='note'),

    path('progress/<int:user_id>/', exp.progress, name='progress'),
    path('img_corner/<int:user_id>/', display_img.corner, name='img_corner'),
    path('img_lr/<int:user_id>/', display_img.lr, name='img_lr'),
    path('submit/<int:user_id>/', display_img.submit, name='submit'),
]
