# Face classification and detection

Real-time face detection & emotion/gender classification using keras CNN model and openCV.
posture judge using SVM and HOG features
face matching by calculating Euclidean distance, using facenet

## Examples
Emotion
Gender
Posture
Matching

## Instructions

 
 
##PlanA


### Run demos:

> python3 video_emotion_color_demo.py

> python3 image_gradcam_demo.py

> python3 video_posture_detect_demo.py

> python3 image_emotion_gender_demo.py <image_path>

e.g.

> python3 image_emotion_gender_demo.py ../images/test_img.jpg

### Demo for the face-recog contest ( remain to be developed )

> python3 video_face_info_detect.py   


### To train previous/new models for emotion classification:

* Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
* Move the downloaded file to the datasets directory inside this repository.
* Untar the file:
* Run the train_emotion_classification.py file
> python3 train_emotion_classifier.py

### To train previous/new models for gender classification:
* Download the imdb_crop.tar file from [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) (It's the 7GB button with the tittle Download faces only).
* Move the downloaded file to the datasets directory inside this repository.
* Untar the file:
> tar -xfv imdb_crop.tar
* Run the train_gender_classification.py file
> python3 train_gender_classifier.py



 
##PlanB
 
Based on faceai

 
###Demo for the face-recog contest:

> python3 fr.py




