# Generated by Django 4.1.2 on 2022-10-19 01:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='use_system',
            field=models.BooleanField(default=True),
        ),
    ]