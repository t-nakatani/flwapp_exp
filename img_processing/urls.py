from django.urls import path
from img_processing.views import basic, exp


urlpatterns = [
    path('register_img/', exp.register_img, name='register_img'),

    path('', basic.home, name='home'),
    path('note/', basic.note, name='note'),

    path('progress/<int:user_id>', basic.progress, name='progress'),

]