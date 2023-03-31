'''
MTCNN — это каскад свёрточных нейронных сетей. В модели используются 3 сети: P-Net, R-Net и O-net. 
Первая P-Net на выходе выдаёт координаты ограничивающих прямоугольников предполагаемых лиц. 
Далее R-net отсекает области, где лиц скорее всего нет и добавляет уровень достоверности к тем областям, которые остались. 
В третьей сети мы снова избавляемся от прямоугольников с низким уровнем достоверности и добавляем координаты пяти лицевых ориентиров.
'''

import cv2
from mtcnn import MTCNN

# захватываем снимок с записи вебки
cap = cv2.VideoCapture(0) 
ret, frame = cap.read()
if cap.isOpened():
    _,frame = cap.read()
    cap.release() 
    if _ and frame is not None:
        cv2.imwrite('recognition/images/latest.jpg', frame)

# ищем на этом снимке лицо и лицевые точки с координатами
detector = MTCNN()
image = cv2.cvtColor(cv2.imread('recognition/images/latest.jpg'), cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']
cv2.rectangle(image,
          	(bounding_box[0], bounding_box[1]),
       	   (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
          	(0,155,255), 2)
cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
cv2.imwrite("sveta.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print(result)