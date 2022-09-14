from cgitb import enable
import re
import cv2
import mediapipe as mp
import time 

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

prevTime = 0
curTime = 0

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, ln in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, ln)
            h, w, c = img.shape
            cx, cy = int(ln.x*w), int(ln.y*h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        

    
    
    curTime = time.time()
    fps = 1/(curTime-prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break