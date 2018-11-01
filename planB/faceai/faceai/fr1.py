#coding=utf-8
# 拍摄操作员
import cv2
import datetime
import time
import os
import numpy as np
#from matplotlib import pyplot as plt


sizeThreshold=200

path = "img/face_recognition"
total=os.listdir(path)
fileNum = len(total)
frameOrigin = np.zeros((1080,1920),np.uint8)#生成一个空灰度图像
cap = cv2.VideoCapture(1)
cap.set(3,1920) #
cap.set(4,1080)
#startTime = datetime.datetime.now()
startTime=time.time()
takeTime=[10,15,20,25]
runTime=round(time.time()-startTime,2)
waitTime=round(time.time()-startTime-takeTime[0],2)
while (fileNum<3):
    ret, frame = cap.read()
    total=os.listdir(path)
    fileNum = len(total)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换灰色

    runTime=round(time.time()-startTime,2)
    waitTime=round(time.time()-startTime,2)
    text=str(runTime)

    cv2.putText(frame,"Time "+text, (10,100),cv2.FONT_HERSHEY_COMPLEX, 2.0, (0,0, 255), 2)#FONT_HERSHEY_SIMPLEX
    classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    #img2 = np.zeros((img.shape[0],img.shape[1],3), np.uint8) 
    color = (0, 255, 0)  # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects)==1:  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            roi_color = frame[y:y+h, x:x+w]
            # 框出人脸
            y=y-h/4
            h=h+h/4
            cv2.rectangle(frame, (x, y), (x + w, y +h), color, 2)
            waitTime=round(time.time()-startTime-takeTime[0],2)
            #text=str(runTime)
            
            if runTime<takeTime[0]:
                waitTime=round(time.time()-startTime-takeTime[0],2)
                text=str(waitTime)
                cv2.putText(frame,"First photo,"+text, (x,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,255, 0), 1)
            elif runTime<takeTime[1] and fileNum==0 :
                waitTime=round(time.time()-startTime-takeTime[0],2)
                text=str(waitTime)
                cv2.imwrite('img/face_recognition/operator'+str(fileNum+1)+'.jpg',frame[y:y+h, x:x+w])
                #cv2.imwrite('img/face_recognition/operator.jpg',frame)#[y:y+h, x:x+w])
                cv2.putText(frame,"OK "+text, (x + 6,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,255, 0), 1)
                print('1: '+text)
                cv2.imshow("image", frame)
                time.sleep(0.5)
            elif runTime<takeTime[2] and runTime>takeTime[1] and fileNum==1:
                waitTime=round(time.time()-startTime-takeTime[0],2)
                text=str(waitTime)
                cv2.imwrite('img/face_recognition/operator'+str(fileNum+1)+'.jpg',frame[y:y+h, x:x+w])
                #cv2.imwrite('img/face_recognition/operator.jpg',frame)#[y:y+h, x:x+w])
                cv2.putText(frame,"OK "+text, (x + 6,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,255, 255), 1)
                print('2: '+text)
                cv2.imshow("image", frame)
                time.sleep(0.5)
            elif runTime<takeTime[3] and runTime>takeTime[2] and fileNum==2:
                waitTime=round(time.time()-startTime-takeTime[0],2)
                text=str(waitTime)
                cv2.imwrite('img/face_recognition/operator'+str(fileNum+1)+'.jpg',frame[y:y+h, x:x+w])
                cv2.putText(frame,"OK "+text, (x + 6,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,0, 0), 1)
                print('3: '+text)
                cv2.imshow("image", frame)
                time.sleep(0.5)
            elif fileNum==3:
                #text=str(runTime)
                cv2.putText(frame,"All Photo Taken", (x + 6,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,0, 0), 1)
                break
            
    cv2.imshow("image", frame)  # 显示图像
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #cv2.imwrite('./img/face_recognition/operator'+str(n)+'.png',frame[])
        break
#'''



# 程序结束时间
endTime = time.time()
print((endTime - startTime))
cap.release()
cv2.destroyAllWindows()

#cv2.putText(img, 'opencv', (10, 500), font, 4, (255, 255, 0), 1, False)　