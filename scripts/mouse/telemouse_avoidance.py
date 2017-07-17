#!/usr/bin/env python
import rospy
from std_msgs.msg import Int16
from std_srvs.srv import Empty

import wiringpi2

import signal
import sys

import os, sys
import numpy as np

import pygame

class Avoidance(object):

    def __init__(self, node_name="telemouse_avoidance"):

        rospy.init_node(node_name)
        sub = rospy.Subscriber("disconfort", Int16, self.callback)
      
        pygame.mixer.init()
        self.hit_sound = pygame.mixer.Sound("n_418d125.wav")

        signal.signal(signal.SIGINT, self.exit_handler)

        self.gp_out = 18
        wiringpi2.wiringPiSetupGpio()
        wiringpi2.pinMode(self.gp_out, wiringpi2.GPIO.PWM_OUTPUT)
        wiringpi2.pwmSetMode(wiringpi2.GPIO.PWM_MODE_MS)
        wiringpi2.pwmSetClock(375)

        self.RIGHT = 56
        self.CENTER = 76
        self.LEFT = 96

        self.rest_counter = 0

    def exit_handler(self, signal, frame):
        print ("\nExit")
        wiringpi2.pwmWrite(self.gp_out, self.CENTER)
        wiringpi2.delay(500)
        sys.exit(0)


    def callback(self, disconfort):
        if disconfort.data == 1:
            """
            if self.rest_counter < 200:
                return

            rospy.wait_for_service("serv_off")
            try:
                soff = rospy.ServiceProxy("serv_off", Empty)
                soff()
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e
                
                
            self.hit_sound.play()
            """
            wiringpi2.pwmWrite(self.gp_out, self.LEFT)
            wiringpi2.delay(500)
            wiringpi2.pwmWrite(self.gp_out, self.RIGHT)
            wiringpi2.delay(500)
            wiringpi2.pwmWrite(self.gp_out, self.CENTER)
            wiringpi2.delay(500)

            try:
                son = rospy.ServiceProxy("serv_on", Empty)
                son()
            except rospy.ServiceException, e:
                print "Service call failed: %s" % e

            self.rest_counter = 0

        else:
            self.rest_counter += 1

if __name__ == "__main__":
    obj = Avoidance()
    rospy.spin()

