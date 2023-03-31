import cv2
import mediapipe as mp
import time 

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2) 
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, face, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)


    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break