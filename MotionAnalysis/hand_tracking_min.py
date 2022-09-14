import cv2
import mediapipe as mp
import time 

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
curTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id == 4: #большой палец
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                #все точки
                # cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED) 
    
            mpDraw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
    
    curTime = time.time()
    fps = 1/(curTime-prevTime)
    prevTime = curTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break