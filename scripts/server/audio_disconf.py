#!/usr/bin/env python
import rospy
from audio_common_msgs.msg import AudioData
from std_msgs.msg import Int16

import os, sys
import numpy as np

# from esn import ESN
from esn_disconf import ESN
import random
import pickle
from scipy.fftpack import fft, fftfreq
import time
import pygame

class AudioDisconf(object):

    def __init__(self, node_name, esn_pickle,
                 acc_t, disconf_t, decay):
        self.N = 256
        self.accum = 0
        self.acc_t = acc_t
        self.disconf_t = disconf_t
        self.decay = 0.6
        rospy.init_node(node_name)        
        with open(esn_pickle, "rb") as f:
            self.esn = pickle.load(f)
        
        sub = rospy.Subscriber("audio", AudioData, self.callback)
      
        self.pub = rospy.Publisher("disconfort", Int16, queue_size=1)
        self.disconf = Int16()

        pygame.mixer.init()
        # self.hit_sound = pygame.mixer.Sound("pi.wav")

    def callback(self, audio):
        data = np.array([ord(n) for n in audio.data[:self.N]], np.float32)/255.
        yf = fft(data)[:self.N/2:4].real/(self.N/2)
        out = self.esn.prop(yf)[1][0]
        self.accum = (self.accum + out) if out > self.disconf_t else self.accum
        if self.accum > self.acc_t:

            print "-------------", os.getcwd()
            rospy.loginfo("disconf")
            print "deconf: ", self.accum
            self.accum = 0
            #self.hit_sound.play()
            self.disconf.data = 1
            self.pub.publish(self.disconf.data)
        else:
            self.disconf.data = 0
            self.pub.publish(self.disconf.data)
            print out
        self.accum *= self.decay
    
    
if __name__ == "__main__":
    obj = AudioDisconf(node_name="audio_disconf", 
                       esn_pickle="/home/osawa/ros_catkin_ws/src/telemouse_v2/scripts/server/esn2.pickle",
                       acc_t=1.5, disconf_t=0.95, decay=0.3)
    rospy.spin()

