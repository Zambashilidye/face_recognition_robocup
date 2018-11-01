#!/usr/bin/env python
#coding:utf-8
from audio import AudioControl 
import os
import time
#import winsound

def DetectImage():
    #print(os.path.abspath("facereg"))
    return os.path.isfile("/home/snfgto/robocup/facereg/face_recog_v1.0/outputs/operator.jpg")

def AutoControl():
    detectflag = 0
    while (detectflag == 0):
        time.sleep(1)
        if DetectImage():
            detectflag = 1
            #winsound.Beep(600,1000)
            #print("\a")
            beep()
            print("Get Image.Now turning.")
            ac=AudioControl("ac")
            ac.turn()
        else:
            detectflag = 0
            print("Don't find the image.Keep searching...")

def beep():
    duration = 1  # second
    freq = 1000  # Hz
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))

if __name__ == '__main__':
    try:
        #rate = 20
        #SetRate(rate)
        AutoControl()
        #print("\a")
    except:
        print("Processes End.")