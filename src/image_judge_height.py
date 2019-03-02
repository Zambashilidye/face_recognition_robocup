import cv2
import numpy as np
from utils.inference import detect_faces
import argparse
from statistics import mode
from keras.models import load_model
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input
import math

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

#parser = argparse.ArgumentParser(description='img')
#parser.add_argument('--img', type=str, default = '../outputs/1.jpg')
#args = parser.parse_args()
video_capture = cv2.VideoCapture(0)
<<<<<<< HEAD
=======
video_capture.get(CV_CAP_PROP_FPS)
>>>>>>> js

while True:
    bgr_image= video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image) 
    img_shape=bgr_image.shape



    def detect_postures(x,y,w,h,img_shape):
        
        s_face=w*h
        s_img=img_shape[0]*img_shape[1]
        
        min_d=0.25 #distance when face can cover the whole img
        alpha=0.53 #camera angle
        base_h=0.5 #camera base height
        
        dif=(img_shape[1]-y)/img_shape[1]

        face_d=np.sqrt(s_img/s_face)*min_d
        #print(math.sin(alpha))
<<<<<<< HEAD
        face_h=face_d/math.tan(alpha)*dif*+base_h
=======
        face_h=face_d/math.tan(alpha)*dif+base_h
>>>>>>> js
        result="unknown"
        print(face_h)
        if face_h>1.47:
            result="stand"
        else:
            result="sit"
        return result


    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, (0,0))
        x=x1
        y=y1
        w=x2-x1
        h=y2-y1
        draw_text(face_coordinates, bgr_image, detect_postures(x,y,w,h,img_shape),
                  (0,0,0), 0, -20, 1, 1)
        #print(detect_postures(x,y,w,h,img_shape))
    cv2.imshow("detect",bgr_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()