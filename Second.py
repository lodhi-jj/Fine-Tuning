# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:30:51 2019

@author: lodhi
"""

# import the necessary packages
from keras.preprocessing.image import img_to_array
from imutils.video import VideoStream
from keras.models import load_model
import numpy as np
import imutils
import time
import cv2

# define the paths to the deep learning model
MODEL_PATH = "da_last4_layers_for_6_Classes.h5"

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    # frame = cv2.imread('clean-dataset/validation/20/20_ (1).jpg')
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    
	# classify the input image and initialize the label and
	# probability of the prediction
    prob = model.predict(image)[0]
    
    
    if prob[0] == max(prob):
        label = 'Pepper__bell___Bacterial_spot'
    elif prob[1] == max(prob):
        label = 'Pepper__bell___healthy'
    elif prob[2] == max(prob):
        label = 'Potato___Early_blight'
    elif prob[3] == max(prob):
        label = 'Potato___healthy'
    elif prob[4] == max(prob):
        label = 'Potato___Late_blight'
    elif prob[5] == max(prob):
        label = 'Plant Not Detected'
    
    
    # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, max(prob) * 100)
    frame = cv2.putText(frame, label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    
# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()