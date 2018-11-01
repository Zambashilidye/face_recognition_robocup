#!/usr/bin/env python
#coding:utf-8
import rospy
#from std_msgs.msg import String
import speech_recognition as sr 
#import re   
from geometry_msgs.msg import Twist
#from sensor_msgs.msg import LaserScan
from math import pi
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
#import difflib
#import os
import time

class AudioControl():

    def __init__(self,objectname):
        self.command = "0"
        self.audioflag = 1
        self.changemodeflag = 0
        rospy.init_node(objectname, anonymous=False)
        #rospy.on_shutdown(self.shutdown)
        self.cmd_vel = rospy.Publisher('/cmd_vel_mux/input/navi',Twist,queue_size=1)
        #rospy.Subscriber('scan',LaserScan,callback)
        self.name1=objectname
        #self.name2=nodename

    def printstring(self):
        print(self.name1)
        #print(self.name2)

    def SetRate(self,rate):
        # Set the equivalent ROS rate variable
        r = rospy.Rate(rate)

    def GetAudio(self):
        if self.audioflag:
            #obtain audio from the microphone 
            r = sr.Recognizer() 
            mic = sr.Microphone()
            with mic as source:
                print("Waiting for command!")
                r.adjust_for_ambient_noise(source,duration=0.7)
                audio = r.listen(source)
                #global command
                self.command=r.recognize_sphinx(audio)
                print(self.command)
            if self.changemodeflag:
                self.ChangeMode()
            else:
                self.BeginChangeMode()

    def BeginChangeMode(self):
        print("start")
        #global command
        string1 = "practice"
        string2 = "complete"
        searchObj1 = fuzz.partial_ratio(self.command,string1)
        print(searchObj1)
        searchObj2 = fuzz.partial_ratio(self.command,string2)
        print(searchObj2)
        #global changemodeflag
        if searchObj1>=60:
            self.changemodeflag = 1
            self.command = ""
            self.ChangeMode()
        elif searchObj2>=60:
            self.changemodeflag = 0
            self.complete()
        else:
            self.changemodeflag = 0
            self.GetAudio()

    def ChangeMode(self):
        print("active")
        #global command
        #match audio with string
        string1="move"
        string2="turn back"
        string3="record"
        string4="search"
        string5="okay"
        searchObj1 = fuzz.partial_ratio(self.command,string1)
        print(searchObj1)
        searchObj2 = fuzz.partial_ratio(self.command,string2)
        print(searchObj2)
        searchObj3 = fuzz.partial_ratio(self.command,string3)
        print(searchObj3)
        searchObj4 = fuzz.partial_ratio(self.command,string4)
        print(searchObj4)
        searchObj5 = fuzz.partial_ratio(self.command,string5)
        print(searchObj5)

        #change mode according to results
        commandslist = [searchObj1,searchObj2,searchObj3,searchObj4,searchObj5]
        commands=[self.move,self.turn,self.record,self.search,self.okay,self.passon]
        num = 5
        if max(commandslist)<40:
            num = 5
        else:
            num = commandslist.index(max(commandslist))
        print("command:" + str(num+1))
        if num>=0:
            #for i in range(10):
            commands[num]()
            self.GetAudio()
        else:
            print("Nothing recognized.")

    def move(self):
        print("move")
        #global audioflag
        self.audioflag = 1
        move_cmd = Twist()
        move_cmd.linear.x = 0.1
        move_cmd.angular.z = 0
        for i in range(10):
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(0.5)

    def turn(self):
        print("turn back")
        #global audioflag
        self.audioflag = 1
        angular_speed = 1.0 
        #goal_angle = pi
        move_cmd = Twist()
        move_cmd.linear.x = 0
        move_cmd.angular.z = angular_speed
        # Rotate for a time to go 180 degrees
        #ticks = int(goal_angle)
        for t in range(10):           
            self.cmd_vel.publish(move_cmd)
            rospy.sleep(0.5)

    def record(self):
        print("recording...")
        #global audioflag
        self.audioflag = 1

    def search(self):
        print("searching...")
        #global audioflag
        self.audioflag = 1

    def complete(self):
        print("Stop.")
        #global audioflag
        self.audioflag = 0
        #rospy.loginfo("Stop.")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)

    def passon(self):
        pass

    def okay(self):
        #global changemodeflag 
        self.changemodeflag = 0
        self.GetAudio()

    def shutdown(self):
        # Always stop the robot when shutting down the node.
        print("Stopping the process...")
        self.cmd_vel.publish(Twist())
        rospy.sleep(1)
    
    
if __name__ == '__main__':
    try:
        #rate = 20
        #SetRate(rate)
        #GetAudio()
        audiocontrol=AudioControl("Audiocontrol")
        audiocontrol.GetAudio()
        #audiocontrol.printstring()
    except:
        print("Processes End.")
    