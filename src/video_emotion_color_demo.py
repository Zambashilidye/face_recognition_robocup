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


image_size = 200 #don't need equal to real image size, but this value should not small than this
modeldir = '20170512-110547/20170512-110547.pb' #change to your model dir

image_name1 = '1.jpg' #change to your image name

 
print('建立facenet embedding模型')
tf.Graph().as_default()
sess = tf.Session()
 
facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
 
print('facenet embedding模型建立完毕')
 
scaled_reshape = []
 
image1 = scipy.misc.imread(image_name1, mode='RGB')
image1 = cv2.resize(image1, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
image1 = facenet.prewhiten(image1)







# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
c=0
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)
    #sc=sc+1
   # distance = compare2face(bgr_image)
    #distance2 = compare2face(rgb_image)
    
   # threshold = 1.10    # set yourself to meet your requirement 
    #print("distance = "+str(distance))
    #print("distance = "+str(distance2))
    #print("Result = " + ("same person" if distance <= threshold else "not same person"))
    
    c=c+1
    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]

        image2=rgb_image[y1:y2,x1:x2]
        #cv2.imshow('window_frame2', image2)
        #cv2.imwrite('2.jpg',image2)
        
        #image2 = scipy.misc.imread("2.jpg", mode='RGB')
        #image2 = cv2.resize(image2, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        if c>10 and image2.shape[0]>0:
            c=0
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
            scaled_reshape.clear()
            print(dist)
            
                

        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
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
        
        if dist<0.8:
            color=(0,0,0)

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)
        #cv2.imshow('window_frame2', gray_face)
    
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
