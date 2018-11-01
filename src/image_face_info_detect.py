#coding:utf-8
from statistics import mode

import cv2
from keras.models import load_model
import numpy as np

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input







import tensorflow as tf
import scipy.misc
import facenet
import time
import argparse
import pygame


############FLAGS#############
if_find_op=0  #寻找标志
op_sample=0   #采样标志
sample_time=0 #采样次数
detect_time=0 #寻找次数
num_of_obj=0  #发现人数
max_of_obj=0  #最多发现人数
end_detect=0  #结束
retry=0       #回退人数
##############################


############MATCHINIT#############
image_size = 200 #don't need equal to real image size, but this value should not small than this
modeldir = '../downloaded_models/20170512-110547/20170512-110547.pb' #change to your model dir

image_name1 = '../outputs/operator.jpg' #change to your image name
print('build facenet embedding model')
tf.Graph().as_default()
sess = tf.Session()
facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
print('facenet embedding building complete!')
scaled_reshape = []
##################################


#image1 = scipy.misc.imread(image_name1, mode='RGB')
#image1 = cv2.resize(image1, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
#image1 = facenet.prewhiten(image1)








############JUDGEINIT#############

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
gender_model_path = '../trained_models/gender_models/simple_CNN.81-0.96.hdf5'
emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
font = cv2.FONT_HERSHEY_SIMPLEX

# hyper-parameters for bounding boxes shape
frame_window = 10
gender_offsets = (30, 60)
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
gender_classifier = load_model(gender_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]
gender_target_size = gender_classifier.input_shape[1:3]

# starting lists for calculating modes
gender_window = []
emotion_window = []

########################################

#############HOG CASCADE INIT###################
HOGCascade = cv2.HOGDescriptor()
HOGCascade.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

winStride = (8,8)
padding = (16,16)
scale = 1.05
meanshift = 0
################################################


#############MAINFUNC###################


# starting video streaming
parser = argparse.ArgumentParser(description='image select')
parser.add_argument('--img1', type=str, default = "../outputs/1.jpg")
parser.add_argument('--img2',type=str, default = "../outputs/4.jpg")
args = parser.parse_args()

soundpath=r'../sound/dendendong.mp3'
pygame.mixer.init()
print("sound initialized!")
track = pygame.mixer.music.load(soundpath)
pygame.mixer.music.play()
time.sleep(3)
pygame.mixer.music.stop()
bgr_image = cv2.imread(args.img2)
gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
faces = detect_faces(face_detection, gray_image)    
status=""

img_op=cv2.imread(args.img1)
gray_img = cv2.cvtColor(img_op, cv2.COLOR_BGR2GRAY)

faces_op=detect_faces(face_detection,gray_img)
for face_coordinates in faces:
    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    img3=img_op[y1:y2,x1:x2]
    #cv2.imshow("1",img3)
    #cv2.waitKey(3)
    #cv2.imwrite('../outputs/operator.jpg',img3)
    
                



for face_coordinates in faces:
    
    x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]    
    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]
    dist=1
    image2=rgb_image[y1:y2,x1:x2]
    Width=int(abs(x2-x1)/1.6)
    Height=abs(y2-y1)
    gray_body=gray_image[y1:y1+10*Height,x1-Width:x2+Width]
    bgr_body=bgr_image[y1:y1+10*Height,x1-Width:x2+Width]

    (rects, weights) = HOGCascade.detectMultiScale(gray_body, winStride=winStride,
                                        padding=padding,
                                        scale=scale,
                                        useMeanshiftGrouping=meanshift)
    status="sit"
    for(x, y, w, h) in rects:
        
        
        if w>0 and (w*h)>0.3*(gray_body.shape[0]*gray_body.shape[1]):
            status="stand"
            print(x,y,w,h)
            cv2.rectangle(bgr_body, (x, y), (x+w, y+h), (0,200,255), 2)
            cv2.imwrite("../outputs/peoplestand.jpg",bgr_body)
            
    
    
    if image2.shape[0]>0:
        image1 = scipy.misc.imread(image_name1, mode='RGB')
        image1 = cv2.resize(image1, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        image1 = facenet.prewhiten(image1)
        #image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
        image2 = cv2.resize(image2, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        image2 = facenet.prewhiten(image2)
        scaled_reshape.append(image1.reshape(-1,image_size,image_size,3))
        emb_array1 = np.zeros((1, embedding_size))
        emb_array1[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False })[0]
        scaled_reshape.append(image2.reshape(-1,image_size,image_size,3))
        emb_array2 = np.zeros((1, embedding_size))
        emb_array2[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False })[0]
    
        dist = np.sqrt(np.sum(np.square(emb_array1[0]-emb_array2[0])))
        scaled_reshape=[]
        print(dist)
        time.sleep(0.3)


    try:
        rgb_face = cv2.resize(rgb_face, (gender_target_size))
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue
    gray_face = preprocess_input(gray_face, False)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_prediction = emotion_classifier.predict(gray_face)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion_text = emotion_labels[emotion_label_arg]
    emotion_window.append(emotion_text)
    emotion_probability = np.max(emotion_prediction)
    rgb_face = np.expand_dims(rgb_face, 0)
    rgb_face = preprocess_input(rgb_face, False)
    
    gender_prediction = gender_classifier.predict(rgb_face)
    gender_label_arg = np.argmax(gender_prediction)
    gender_text = gender_labels[gender_label_arg]
    gender_window.append(gender_text)

    if len(gender_window) > frame_window:
        emotion_window.pop(0)
        gender_window.pop(0)
    try:
        emotion_mode = mode(emotion_window)
        gender_mode = mode(gender_window)
    except:
        continue

    if emotion_text == 'angry':
        color = emotion_probability * np.asarray((255, 0, 0))
    elif emotion_text == 'sad':
        color = emotion_probability * np.asarray((0, 0, 255))
    elif emotion_text == 'happy':
        color = emotion_probability * np.asarray((255, 255, 0))
    elif emotion_text == 'surprise':
        color = emotion_probability * np.asarray((0, 255, 255))
    else:
        color = emotion_probability * np.asarray((0, 255, 0))

    
    draw_text(face_coordinates, rgb_image, gender_mode,
                color, 0, -20, 1, 1)
    draw_text(face_coordinates, rgb_image, emotion_mode,
                color, 0, -45, 1, 1)
    draw_text(face_coordinates, rgb_image, status,
                color, 100, -20, 1, 1)
    if dist<0.5:
        draw_text(face_coordinates, rgb_image, "operator",
                color, 0, -80, 1, 1)
        draw_bounding_box(face_coordinates, rgb_image, (0,0,0))
        if_find_op=1

    else:

        draw_bounding_box(face_coordinates, rgb_image, color)

    

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
cv2.imwrite("../outputs/result.jpg",bgr_image)

print("detect finished!")
pygame.mixer.music.play()
time.sleep(3)
pygame.mixer.music.stop()