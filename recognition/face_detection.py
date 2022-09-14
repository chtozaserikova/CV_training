import cv2
import mediapipe as mp
import time 

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.9) #чем выше, тем меньше FalseePositive

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            bb_white = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bb_pink = int(bb_white.xmin*iw), int(bb_white.ymin*ih), int(bb_white.width*iw), int(bb_white.height*ih)
            cv2.rectangle(img, bb_pink, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (bb_pink[0], bb_pink[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        

    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break