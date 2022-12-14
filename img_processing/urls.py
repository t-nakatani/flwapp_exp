from django.urls import path
from img_processing.views import basic, exp, use_system, manual


urlpatterns = [
    path('register_img/', exp.register_img.as_view(), name='register_img'),

    path('', basic.home, name='home'),
    path('note/', basic.note, name='note'),
    path('agree/', basic.agree, name='agree'),

    # 動作確認用の試用環境
    path('trial/<int:user_id>/', exp.trial_env, name='trial'),

    path('progress/<int:user_id>/', exp.progress, name='progress'),
    path('select_arrangement/<int:user_id>/', manual.arrangement, name='select_arrangement'),
    path('submit/manual/<int:user_id>/<str:predict>/', manual.submit, name='manual_submit'),
    path('img_corner/<int:user_id>/', use_system.corner, name='img_corner'),
    path('img_lr/<int:user_id>/', use_system.lr, name='img_lr'),
    path('submit/system/<int:user_id>/', use_system.submit, name='use_system_submit'),
    path('bug_report/<int:user_id>/', exp.bug_report_for_exp, name='bug_report'),
    path('questionnaire/<int:user_id>/', exp.questionnaire, name='questionnaire'),
]
