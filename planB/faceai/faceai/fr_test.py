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
from autoadjust import AutoAdjust,detect_postures


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
goal_angle = 2.75
angular_duration = goal_angle / angular_speed
move_cmd=Twist()
cmd_vel.publish(move_cmd)


r = rospy.Rate(rate)

os.system("paplay 1.wav")


##-------------------------------
## in[1] 记忆
if os.path.exists("img/face_recognition/operator1.jpg"):
    os.remove("img/face_recognition/operator1.jpg")
if os.path.exists("img/face_recognition/operator2.jpg"):
    os.remove("img/face_recognition/operator2.jpg")
if os.path.exists("img/face_recognition/operator3.jpg"):
    os.remove("img/face_recognition/operator3.jpg")
if os.path.exists("img/face_recognition/operator.jpg"):
    os.remove("img/face_recognition/operator.jpg")
sizeThreshold=200
path = "img/face_recognition"
total=os.listdir(path)
fileNum = len(total)
print(fileNum)
frameOrigin = np.zeros((1080,1920),np.uint8)#生成一个空灰度图像
cap = cv2.VideoCapture(0)
cap.set(3,1920) #
cap.set(4,1080)
#startTime = datetime.datetime.now()
startTime=time.time()
takeTime=[10,15,20,30,60]
runTime=round(time.time()-startTime,2)
waitTime=round(time.time()-startTime-takeTime[0],2)
while (fileNum<3):
    ret, frame = cap.read()
    total=os.listdir(path)
    fileNum = len(total)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换灰色

    runTime=round(time.time()-startTime,2)
    waitTime=round(takeTime[fileNum]-runTime,2)
    text=str(runTime)

    cv2.putText(frame,"Time "+text, (10,100),cv2.FONT_HERSHEY_COMPLEX, 2.0, (0,0, 255), 2)#FONT_HERSHEY_SIMPLEX
    classifier = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    #img2 = np.zeros((img.shape[0],img.shape[1],3), np.uint8) 
    color = (0, 255, 0)  # 定义绘制颜色
    # 调用识别人脸
    faceRects = classifier.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    if len(faceRects)==1:#and runTime<takeTime[4]:  # 大于0则检测到人脸
        #print(faceRects)
        #print(type(faceRects))
        x, y, w, h = faceRects[0]
        roi_color = frame[y:y+h, x:x+w]
        # 框出人脸
        y=y-h/4
        h=h+h/3
        cv2.rectangle(frame, (x, y), (x + w, y +h), color, 2)
        waitTime=round(time.time()-startTime-takeTime[0],2)
        #text=str(runTime)
        if runTime<takeTime[fileNum]:
            runTime=round(time.time()-startTime,2)
            waitTime=round(takeTime[fileNum]-runTime,2)
            text=str(waitTime)
            cv2.putText(frame,"Wait "+text, (x,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0,0,0), 1)
        elif runTime<takeTime[fileNum+1] and runTime>takeTime[fileNum] and fileNum<3:
            runTime=round(time.time()-startTime,2)
            waitTime=round(takeTime[fileNum]-runTime,2)
            #text=str(waitTime)
            cv2.imwrite('img/face_recognition/operator'+str(fileNum+1)+'.jpg',frame[y:y+h, x:x+w])
            cv2.putText(frame,"Second "+str(waitTime), (x + 6,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (255,255, 0), 1)
            print(str(fileNum+1)+" "+text)
            cv2.imshow("image",frame)
            time.sleep(0.5)
        elif fileNum<3:
            #text=str(runTime)
            cv2.imwrite('img/face_recognition/operator'+'.jpg',frame[y:y+h, x:x+w])
            cv2.putText(frame,"Bad", (x + 6,y - 6), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255,0, 0), 1)
            break
    elif runTime<takeTime[4]:
        pass
    else:
        break
    cv2.imshow("image", frame)  # 显示图像
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #cv2.imwrite('./img/face_recognition/operator'+str(n)+'.png',frame[])
        break



# 程序结束时间
endTime = time.time()
print((endTime - startTime))

cap.release()
cv2.destroyAllWindows()



os.system("paplay 1.wav")
time.sleep(10)
########################################################
## in[2] 转动

move_cmd = Twist()
rospy.sleep(1)

move_cmd.angular.z = angular_speed
ticks = int(goal_angle * rate)*2
print("ticks:"+str(ticks))
for t in range(ticks):           
    cmd_vel.publish(move_cmd)
    r.sleep()
        
    # Stop the robot before the next leg

    #cmd_vel.publish(move_cmd)  
    
# Stop the robot
cmd_vel.publish(Twist())


#######################################################
##in[3]  识别  ##
if os.path.exists("photo.jpg"):
    os.remove("photo.jpg")
if os.path.exists("seu.jpg"):
    os.remove("seu.jpg")

startTime=time.time()
path = "img/face_recognition"  # 模型数据图片目录
cap = cv2.VideoCapture(0)
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
time.sleep(1)
#runNum=0
flag=0
runflag=0
turnflag=0
operatorNum=0
faceNum=0
lastfaceNum=0
lastlastfaceNum=0
runNum=0
while(not flag):
    cv2.waitKey(300)
    runNum=runNum+1
    operatorNum=0
    #ret, frame = cap.read()
    #time.sleep(1)
    ret,frame=cap.read()
    cv2.imwrite('photo.jpg',frame)
    #cap.release()
    
    frame=cv2.imread('photo.jpg')

    faceNum=0
    #cv2.imshow('frame',frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Find operator
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    #find the maximum and minimum left
    right_max = 0
    left_min = 0
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        if right_max == 0:
            right_max=right
        elif right_max<right:
            right_max=right
        if left_min == 0:
            left_min=left
        elif left_min>left:
            left_min=left

    AutoAdjust(cmd_vel,int(left_min),int(right_max),turnflag)

    
    for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings):
        faceNum=faceNum+1
        y=  top
        x=  left
        w=  right-left
        h=  bottom-top
        #y+h=    bottom
        #x+w=    right
        img_shape=(1920,1080,3)
        posture=detect_postures(int(x),int(y),int(w),int(h),img_shape)
        
        # 框脸

        gray_face = gray[(top):(bottom+h/10), (left):(right)]
        gray_face = cv2.resize(gray_face, (48, 48))
        gray_face = gray_face / 255.0
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
        emotion = emotion_labels[emotion_label_arg]
        

        # gender
        #print(frame.shape)
        face = frame[top-60:bottom+60, left-60:right+60]
        cv2.imwrite('faceOperator.jpg',face)
        if (bottom-top<=0) or (right-left) <=0 :
            print("too small")
            runflag=1
            #cv2.putText(frame, 'gender', (left+5, bottom+h/10+22 ), cv2.FONT_HERSHEY_TRIPLEX, 1.0,(100, 100, 0), 1)
        else:
            face = cv2.resize(face, (48, 48))
            face = np.expand_dims(face, 0)
            face = face / 255.0
            gender_label_arg = np.argmax(gender_classifier.predict(face))
            gender = gender_labels[gender_label_arg]
            #cv2.putText(frame, gender, (left+5, bottom+h/10+22 ), cv2.FONT_HERSHEY_TRIPLEX, 1.0,(100, 100, 0), 1)
            runflag=0
            #cv2.rectangle(frame, (left, top), (right, bottom),(0,255,255), 2)
        
        
        # operator
        for i, v in enumerate(total_face_encoding):
            match = face_recognition.compare_faces(
                [v], face_encoding, tolerance=0.4)
            name = " "
            if match[0]:
                name = total_image_name[i]
                operatorNum=operatorNum+1
                break

        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255),cv2.FILLED)
        font = cv2.FONT_HERSHEY_TRIPLEX
        

       
        # 标签
        cv2.rectangle(frame, (left, top-h/4), (right, bottom+h/10), (0, 0, 200), 2)
        cv2.rectangle(frame, (left-1, bottom+h/10), (right, bottom+h/10+88), (200, 200, 200),cv2.FILLED)
        
        cv2.putText(frame, emotion, (left+5, bottom+h/10+44), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                    (0, 100, 100), 1)
        if runflag==0:
            cv2.putText(frame, gender, (left+5, bottom+h/10+22 ), cv2.FONT_HERSHEY_TRIPLEX, 1.0,(100, 100, 0), 1)
        cv2.putText(frame, posture, (left+5 , bottom+h/10+66), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                    (0, 0, 100), 1)
        cv2.putText(frame, name, (left+5 , bottom+h/10+110), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                    (0, 0, 100), 1)
        cv2.putText(frame,"yellow", (left+5 , bottom+h/10+88), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                    (0, 0, 100), 1)

    #when False, continue
    flag= ((operatorNum==1) and (lastfaceNum==faceNum) and (runflag==0) and (turnflag==0)) and runNum<100 and lastlastfaceNum==faceNum
    print("operatorNum: "+str(operatorNum))
    print("lastfaceNum: "+str(lastfaceNum))
    print("faceNum:  "+str(faceNum))
    print("flag: "+str(flag))
    #cv2.imshow("11",frame)
    #cv2.waitKey(1)
    lastlastfaceNum=lastfaceNum
    lastfaceNum=faceNum
    

image = face_recognition.load_image_file("photo.jpg")

#查找图像中所有面部的所有面部特征
face_landmarks_list = face_recognition.face_landmarks(image)
#print("face_load ")
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

        cnt=np.array(face_landmarks[facial_feature])
        left=np.min(cnt, axis=0)[0]
        top=np.min(cnt, axis=0)[1]
        right=np.max(cnt, axis=0)[0]
        bottom=np.max(cnt, axis=0)[1]
        if facial_feature=="chin":
            left=(2*left+right)/3  #l+(r-l)/3
            top=(3*top+bottom)/3 #t+(b-t)/2
            right=(2*left+left)/3
            #bottom=np.max(cnt, axis=0)[1]
        else:
            cv2.rectangle(frame,(left-5,top-5),(right+5,bottom+5),(200,200,200),1)
            #if facial_feature=='right_eye':
            #    d.line(face_landmarks[facial_feature], fill=(255, 0, 0), width=2)
            #else:
            #    d.line(face_landmarks[facial_feature], fill=(255, 255, 255), width=2)
        #pil_image.show()
    #x,y,w,h = cv2.boundingRect(cnt)
    #img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    
cv2.imwrite('result/6-SEU-UniRobot.jpg',frame)
cap.release()
endTime=time.time()
print(endTime-startTime)
time.sleep(3)
cv2.destroyAllWindows()
## 提示音
os.system("paplay 1.wav")

