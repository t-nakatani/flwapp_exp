# Generated by Django 4.1.2 on 2022-11-27 10:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0005_alter_user_num_finished_img'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='trial_finished',
            field=models.BooleanField(default=False),
        ),
    ]
