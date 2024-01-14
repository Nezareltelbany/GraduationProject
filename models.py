from django.db import models

# Create your models here.


class RoomMember(models.Model):
    name = models.CharField(max_length=200)
    uid = models.CharField(max_length=200)
    room_name = models.CharField(max_length=200)

    def __str__(self):
        return self.name

'''
1- Create Database Model (RoomMember) | Store user name, uid and room name

2- On Join, create RoomMember in database

3- On HandleUserJoin event, Query db for room member name by uid and room name

4- On leave, delete RoomMember

'''

