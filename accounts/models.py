from django.db import models
from django.contrib.auth.models import AbstractUser

# Create your models here.
class User(AbstractUser):
    """
    ユーザ情報を管理するためのクラス
    """
    next_img_id = models.IntegerField(default=0)

    def __str__(self):
        return self.username
