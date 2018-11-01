#coding=utf-8
#人脸识别类 - 使用face_recognition模块
import cv2
import face_recognition
import os
import datetime
import time
import numpy as np
from keras.models import load_model
import numpy as np
from PIL import Image, ImageDraw


## 机器启动
import rospy
from geometry_msgs.msg import Twist
from math import pi


command = "0"
audioflag = 1
rospy.init_node('Control', anonymous=False)
cmd_vel = rospy.Publisher('/cmd_vel_mux/input/navi',Twist,queue_size=1)
rate = 50
r = rospy.Rate(rate)
angular_speed = 1.0
goal_angle = pi
angular_duration = goal_angle / angular_speed
move_cmd=Twist()
cmd_vel.publish(move_cmd)


r = rospy.Rate(rate)



###################################################
## in[1] 记忆

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
#'''
while (1):
    ret, frame = cap.read()
    total=os.listdir(path)
    fileNum = len(total)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换灰色
    classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    #img2 = np.zeros((img.shape[0],img.shape[1],3), np.uint8) 
    color = (0, 255, 0)  # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    runTime=time.time()-startTime
    if len(faceRects)==1:  # 大于0则检测到人脸
        for faceRect in faceRects:  # 单独框出每一张人脸
            x, y, w, h = faceRect
            roi_color = frame[y:y+h, x:x+w]
            # 框出人脸
            
            cv2.rectangle(frame, (x, y), (x + w, y + 5*h/4), color, 2)
            runTime=time.time()-startTime+5
            text=str(runTime)
            cv2.putText(frame,"Time "+text, (10,100),cv2.FONT_HERSHEY_COMPLEX, 2.0, (0,0, 255), 2)#FONT_HERSHEY_SIMPLEX
            if runTime<10:
                text=str(runTime)
                cv2.putText(frame,"First photo,"+text, (10,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,0, 0), 1)
            elif runTime<15 and fileNum==0 :
                #print
                text=str(runTime)
                #print(file)
                cv2.imwrite('img/face_recognition/operator'+str(fileNum+1)+'.jpg',frame[y:y+h, x:x+w])
                #cv2.imwrite('img/face_recognition/operator.jpg',frame)#[y:y+h, x:x+w])
                cv2.putText(frame,"OK "+text, (x + 6,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,0, 0), 1)
                print('ok')
                cv2.imshow("image", frame)
                time.sleep(0.5)
            elif runTime<20 and fileNum==1:
                #print
                text=str(runTime)
                #print(file)
                cv2.imwrite('img/face_recognition/operator'+str(fileNum+1)+'.jpg',frame[y-h/3:y+h, x-w/3:x+w])
                #cv2.imwrite('img/face_recognition/operator.jpg',frame)#[y:y+h, x:x+w])
                cv2.putText(frame,"OK "+text, (x + 6,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,0, 0), 1)
                print(text)
                cv2.imshow("image", frame)
                time.sleep(0.5)
            elif runTime<25 and fileNum==2:
                #print
                text=str(runTime)
                #print(file)
                cv2.imwrite('img/face_recognition/operator'+str(fileNum+1)+'.jpg',frame[y-h/3:y+h, x-w/3:x+w])
                #cv2.imwrite('img/face_recognition/operator.jpg',frame)#[y:y+h, x:x+w])
                cv2.putText(frame,"OK "+text, (x + 6,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,0, 0), 1)
                print(text)
                cv2.imshow("image", frame)
                time.sleep(0.5)
            else:
                text=str(runTime)
                cv2.putText(frame,"Now "+text, (x + 6,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,0, 0), 1)
            
    cv2.imshow("image", frame)  # 显示图像
    if cv2.waitKey(1) & 0xFF == ord('q') or runTime>30:
        #cv2.imwrite('./img/face_recognition/operator'+str(n)+'.png',frame[])
        break
#'''



# 程序结束时间
endTime = time.time()
print((endTime - startTime))
cap.release()
cv2.destroyAllWindows()


########################################################
## in[2] 转动
for i in range(2):
    move_cmd = Twist()
    rospy.sleep(1)
    
    move_cmd.angular.z = angular_speed
    ticks = int(goal_angle * rate)
    for t in range(ticks):           
        self.cmd_vel.publish(move_cmd)
        r.sleep()
        
    # Stop the robot before the next leg
    move_cmd = Twist()
    cmd_vel.publish(move_cmd)
    rospy.sleep(1)    
    
# Stop the robot
self.cmd_vel.publish(Twist())


#######################################################
##in[3]  识别  ##


startTime=time.time()
path = "img/face_recognition"  # 模型数据图片目录
cap = cv2.VideoCapture(1)
cap.set(3,1920) #
cap.set(4,1080)
total_image_name = []
total_face_encoding = []
for fn in os.listdir(path):  #fn 表示的是文件名q
    print(path + "/" + fn)
    total_face_encoding.append(
        face_recognition.face_encodings(
            face_recognition.load_image_file(path + "/" + fn))[0])
    #fn = fn[:(len(fn) - 4)]  #截取图片名（这里应该把images文件中的图片名命名为为人物名)
    fn='operater'
    total_image_name.append(fn)  #图片名字列表

gender_classifier = load_model(
    "classifier/gender_models/simple_CNN.81-0.96.hdf5")
gender_labels = {0: 'Female', 1: 'Male'}
color = (255, 255, 255)


emotion_classifier = load_model(
    'classifier/emotion_models/simple_CNN.530-0.65.hdf5')


emotion_labels = {
    0: 'Angry',
    1: 'Sick',
    2: 'Terified',
    3: 'Happy',
    4: 'Sad',
    5: 'Suprised',
    6: 'Calm'
}



ret, frame = cap.read()
time.sleep(2)
ret,frame=cap.read()
cv2.imwrite('photo.jpg',frame)
cap.release()

frame=cv2.imread('photo.jpg')

#cv2.imshow('frame',frame)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


#Find operator
face_locations = face_recognition.face_locations(frame)
face_encodings = face_recognition.face_encodings(frame, face_locations)
'''
for (x, y, w, h) in faces:
    face = img[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, 0)
    face = face / 255.0
    gender_label_arg = np.argmax(gender_classifier.predict(face))
    gender = gender_labels[gender_label_arg]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    img = chineseText.cv2ImgAddText(img, gender, x + h, y, color, 30)
'''
# 在这个视频帧中循环遍历每个人脸
for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings):
    
    y=  top
    x=  left
    w=  right-left
    h=  bottom-top
    #y+h=    bottom
    #x+w=    right
    
    # 画出一个框，框住脸
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    # 画出一个带名字的标签，放在框下
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255),cv2.FILLED)

    gray_face = gray[(top):(bottom), (left):(right)]
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]
    cv2.putText(frame, emotion, (left, bottom +20), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                (0, 0, 255), 1)

    ####################




    #print(top,type(top),int(top))
    #print((int(top) - 60),(int(bottom) + 60), (int(right) - 30),(int(left) + 30))
    print(frame.shape)
    #face = frame[(int(top) - 60):(int(bottom) + 60), (int(right) - 30):(int(left) + 30)]
    face = frame[top-60:bottom+60, left-60:right+60]
    cv2.imwrite('facetmp.jpg',face)
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, 0)
    face = face / 255.0
    gender_label_arg = np.argmax(gender_classifier.predict(face))
    gender = gender_labels[gender_label_arg]
    cv2.rectangle(frame, (left, top), (right, bottom),(0,255,255), 2)
    cv2.putText(frame, gender, (left , bottom ), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                (0, 0, 255), 1)
    #img = chineseText.cv2ImgAddText(img, gender, x + h, y, color, 30)
    
    # 看看面部是否与已知人脸相匹配。
    for i, v in enumerate(total_face_encoding):
        match = face_recognition.compare_faces(
            [v], face_encoding, tolerance=0.5)
        name = " "
        if match[0]:
            name = total_image_name[i]
            break

    # 画出一个框，框住脸
    #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    # 画出一个带名字的标签，放在框下
    #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255),cv2.FILLED)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, name, (left , bottom+40), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                (0, 0, 255), 1)


#image = face_recognition.load_image_file("img/ag.png")
image=frame
#查找图像中所有面部的所有面部特征
face_landmarks_list = face_recognition.face_landmarks(image)

for face_landmarks in face_landmarks_list:
    facial_features = [
        'chin',  # 下巴  type(face_landmarks_list[0]['nose_tip'])

        'left_eyebrow',  # 左眉毛
        'right_eyebrow',  # 右眉毛
        'nose_bridge',  # 鼻樑
        'nose_tip',  # 鼻尖
        'left_eye',  # 左眼
        'right_eye',  # 右眼
        'top_lip',  # 上嘴唇
        'bottom_lip'  # 下嘴唇
    ]

    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    for facial_feature in facial_features:
        if facial_feature=='right_eye':
            d.line(face_landmarks[facial_feature], fill=(255, 0, 0), width=2)
        else:
            d.line(face_landmarks[facial_feature], fill=(255, 255, 255), width=2)
    pil_image.show()

os.system("paplay BEEP1.WAV")
cv2.imwrite('seu.jpg',frame)
endTime=time.time()
print(endTime-startTime)
time.sleep(3)
cv2.destroyAllWindows()
## 提示音
os.system("paplay 1.wav")

