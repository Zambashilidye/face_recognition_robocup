#coding:utf-8
import numpy as np
import cv2

cap = cv2.VideoCapture(1)
#cap.set(cv2.CV_CAP_PROP_FRAME_WIDTH,1280)
#cap.set(cv2.CV_CAP_PROP_FRAME_HEIGHT,1080)
#cap.set(3,1920) #
#cap.set(4,1080)
cap.set(3,720) 
cap.set(4,480)
n=0
#cap = cv2.VideoCapture("http://223.3.99.18:4747/video" ) 
#cv2.resize(src,dsize,dst=None,fx=None,fy=None,interpolation=None)
while(True):

    ret,frame = cap.read()
    frame = cv2.GaussianBlur(frame,(5,5),0)
    #frame=cv2.resize(frame,(960,480))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        #s = "%01d"%(n)
        #s = n.zfill(5)
        #cv2.imwrite('./dataset/faces/j'+s+'.jpg',frame)
        cv2.imwrite('./img/face_recognition/operator'+str(n)+'.png',frame)
        n=n+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #cv2.imwrite('./dataset/face0.png',frame)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
