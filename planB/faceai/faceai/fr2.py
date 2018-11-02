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




if os.path.exists("photo.jpg"):
    os.remove("photo.jpg")
if os.path.exists("seu.jpg"):
    os.remove("seu.jpg")

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
# 在这个视频帧中循环遍历每个人脸
for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings):
    
    y=  top
    x=  left
    w=  right-left
    h=  bottom-top
    #y+h=    bottom
    #x+w=    right
    
    # 框脸
    cv2.rectangle(frame, (left, top-h/4), (right, bottom+h/5), (0, 0, 200), 2)
    # 标签
    cv2.rectangle(frame, (left, bottom), (right, bottom+h/5+30), (200, 200, 200),cv2.FILLED)

    gray_face = gray[(top):(bottom), (left):(right)]
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face / 255.0
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion = emotion_labels[emotion_label_arg]
    cv2.putText(frame, emotion, (left+5, bottom +40), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                (0, 100, 100), 1)

    # gender
    print(frame.shape)
    face = frame[top-60:bottom+60, left-60:right+60]
    cv2.imwrite('faceOperator.jpg',face)
    face = cv2.resize(face, (48, 48))
    face = np.expand_dims(face, 0)
    face = face / 255.0
    gender_label_arg = np.argmax(gender_classifier.predict(face))
    gender = gender_labels[gender_label_arg]
    #cv2.rectangle(frame, (left, top), (right, bottom),(0,255,255), 2)
    cv2.putText(frame, gender, (left+5, bottom+20 ), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                (100, 100, 0), 1)
    
    # operator
    for i, v in enumerate(total_face_encoding):
        match = face_recognition.compare_faces(
            [v], face_encoding, tolerance=0.5)
        name = " "
        if match[0]:
            name = total_image_name[i]
            break

    #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 255, 255),cv2.FILLED)
    font = cv2.FONT_HERSHEY_TRIPLEX
    cv2.putText(frame, name, (left+5 , bottom+60), cv2.FONT_HERSHEY_TRIPLEX, 1.0,
                (0, 0, 100), 1)


image = face_recognition.load_image_file("photo.jpg")

#查找图像中所有面部的所有面部特征
face_landmarks_list = face_recognition.face_landmarks(image)
print("face_load ")
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

        cv2.rectangle(frame,(left-5,top-5),(right+5,bottom+5),(200,200,200),1)
        #if facial_feature=='right_eye':
        #    d.line(face_landmarks[facial_feature], fill=(255, 0, 0), width=2)
        #else:
        #    d.line(face_landmarks[facial_feature], fill=(255, 255, 255), width=2)
    #pil_image.show()
#x,y,w,h = cv2.boundingRect(cnt)
#img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imwrite('seu.jpg',frame)
endTime=time.time()
print(endTime-startTime)
time.sleep(3)
cv2.destroyAllWindows()

