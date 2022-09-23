import numpy as np
import cv2
import sys
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)
success = webcam.isOpened()
if success == True:
    # your webcam capabilities
    webcam.set(cv2.CAP_PROP_FPS,30)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH,1024)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,768)
elif success == False:
    print('Error: Camera could not be opened')

while(True):
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Next run filters
    gray = cv2.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    out = frame.copy()
    for (x,y,w,h) in faces:		
        cv2.rectangle(out,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = out[y:y+h, x:x+w]

    cv2.imshow('Face tracker', out)

    if cv2.waitKey(5) == 27:
        break


webcam.release()
cv2.destroyAllWindows()