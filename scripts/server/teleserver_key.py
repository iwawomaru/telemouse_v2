#!/usr/bin/env python

import rospy
from std_msgs.msg import Int16MultiArray
from std_msgs.msg import Int16
import pygame

pygame.init()
from pygame.locals import *
pygame.display.set_mode((10,10), 0, 32)
screen = pygame.display.get_surface()

rospy.init_node("teleserver_key")
pub_dir = rospy.Publisher("direction", Int16MultiArray, queue_size=1)
pub_dconf = rospy.Publisher("disconfort", Int16, queue_size=1)

rate = rospy.Rate(10)
array = Int16MultiArray()
disconf = Int16()

pos = [1, 1]
while not rospy.is_shutdown():
    pygame.display.update()
    array.data = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    f_keydown = False
    disconf.data = 0
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                pos[0] -= 1
            if event.key == K_RIGHT:
                pos[0] += 1
            if event.key == K_UP:
                pos[1] -= 1
            if event.key == K_DOWN:
                pos[1] += 1
            
            if event.key == K_SPACE:
                disconf.data = 1
                rospy.loginfo("user disconf")

        if event.type == KEYUP:
            pos = [1, 1]
            

    array.data[pos[1]*3+pos[0]] = 1

    pub_dir.publish(array)
    pub_dconf.publish(disconf)
    rate.sleep()
    
