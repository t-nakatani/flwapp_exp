from email.policy import default
from django.db import models
from django.contrib.auth.models import AbstractUser
import random


def init_img_list():
    """img_idsのlistをcharで保持"""
    return ','.join([str(i) for i in range(30)])


# Create your models here.
class User(AbstractUser):
    """
    ユーザ情報を管理するためのクラス
    """
    next_img_id = models.IntegerField(default=0)
    use_system = models.BooleanField(default=True)
    char_img_ids = models.CharField(max_length=80, default=init_img_list())  # ','区切りのstrで保持
    num_finished_img = models.IntegerField(default=0)
    trial_finished = models.BooleanField(default=False)  # 操作方法の確認が終わっているか

    def __str__(self):
        return self.username

    def set_next_img_id(self, finished=True):
        """
        view(use_system, manual)のsubmit()で呼ばれる
        char_img_idsをもとに復元したlistから1つidを取り出す
        next_img_idとchar_img_idsとnum_finished_imgを更新
        if finished:
            num_finished_imgとuse_systemを更新
        """
        list_img_ids = (self.char_img_ids).split(',')
        sampled_id = random.choice(list_img_ids)

        self.next_img_id = int(sampled_id)
        list_img_ids.remove(sampled_id)
        self.char_img_ids = ','.join(list_img_ids)
        if finished:
            self.num_finished_img += 1
            if self.num_finished_img == 15:
                self.use_system = not self.use_system
