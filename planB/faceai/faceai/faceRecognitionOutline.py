#coding=utf-8
#绘制面部轮廓
import face_recognition
from PIL import Image, ImageDraw
import numpy as np

# 将图片文件加载到numpy 数组中
image = face_recognition.load_image_file("img/ag.png")

#查找图像中所有面部的所有面部特征
face_landmarks_list = face_recognition.face_landmarks(image)
#face_landmarks_list[0]
for face_landmarks in face_landmarks_list:
    facial_features = [
        'chin',  # 下巴
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
    cnt=np.array(face_landmarks['right_eye'])
    print(cnt)
    left=np.min(cnt, axis=0)[0]
    top=np.min(cnt, axis=0)[1]
    right=np.max(cnt, axis=0)[0]
    bottom=np.max(cnt, axis=0)[1]
    print(left,top,right,bottom)
    #cv2.rectangle(frame,(left,top),(right,bottom),(0,255,255),2)
    d.rectangle([(left,top),(right,bottom)])
    pil_image.show()
