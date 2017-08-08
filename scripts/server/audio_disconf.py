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
        self.num_average = 50
        self.accum = 0
        self.accum_history = [0] * self.num_average
        self.acc_t = acc_t
        self.disconf_t = disconf_t
        self.decay = 0.6
        rospy.init_node(node_name)        
        with open(esn_pickle, "rb") as f:
            self.esn = pickle.load(f)
        
        sub = rospy.Subscriber("audio", AudioData, self.callback)
        sub_key = rospy.Subscriber("disconfort_key", Int16, self.callback_key)

        self.pub = rospy.Publisher("disconfort", Int16, queue_size=1)
        self.disconf = Int16()

        pygame.mixer.init()
        # self.hit_sound = pygame.mixer.Sound("pi.wav")

    def callback(self, audio):
        data = np.array([ord(n) for n in audio.data[:self.N]], np.float32)/255.
        yf = fft(data)[:self.N/2:4].real/(self.N/2)
        out = self.esn.prop(yf)[1][0]
        self.accum = (self.accum + out) if out > self.disconf_t else self.accum
        self.accum_history.append(self.accum)
        # rospy.loginfo("accum value = " + str(self.accum))

        if self.accum > self.acc_t:
            # rospy.loginfo("disconf")
            self.accum = 0
            self.disconf.data = 1
        else:
            self.disconf.data = 0
        self.pub.publish(self.disconf.data)
        self.accum = (self.accum * self.decay) if (self.accum > 0.01) else 0.

    def callback_key(self, data):
        rospy.loginfo("call callback_key %f", 
                      np.sum(self.accum_history[-self.num_average:]))
        target_var = np.average(self.accum_history[-self.num_average:])
        self.acc_t -= 0.4*(self.acc_t - target_var)
        return
    
if __name__ == "__main__":
    """
    # This parameter is well choned
    obj = AudioDisconf(node_name="audio_disconf", 
                       esn_pickle="/home/osawa/ros_catkin_ws/src/telemouse_v2/scripts/server/esn2.pickle",
                       acc_t=1.5, disconf_t=0.95, decay=0.3)
    """

    obj = AudioDisconf(node_name="audio_disconf",
                       esn_pickle="/home/osawa/ros_catkin_ws/src/telemouse_v2/scripts/server/esn2.pickle",
                       acc_t=5.0, disconf_t=0.95, decay=0.3)
    rospy.spin()

