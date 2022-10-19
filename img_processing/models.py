from distutils.command.upload import upload
from django.db import models
from accounts.models import User

# Create your models here.
class Image(models.Model):
    """画像のアップロードに用いるクラス"""
    img_id = models.CharField(max_length=2)
    img = models.ImageField(upload_to='estimated/upload/')


class ImageProcessing(models.Model):
    """
    ユーザが各画像を処理した際の記録を担うクラス
    """
    user = models.ForeignKey(User, on_delete=models.PROTECT)
    img_id = models.IntegerField(default=0)
    predict = models.CharField(max_length=12, default="")
    start_time = models.DateTimeField(auto_now_add=True)  # 登録時に現在時刻で更新
    end_time = models.DateTimeField(auto_now=True)  # 登録時と更新時に現在時刻で更新
