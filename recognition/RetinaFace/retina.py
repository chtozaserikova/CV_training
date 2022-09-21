from retinaface import RetinaFace
import cv2
import os

img_path = 'recognition/images/latest.jpg'
faces = RetinaFace.detect_faces(img_path, threshold = 0.7)
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
{
    "face_1": {
        "score": 0.9993440508842468,
        "facial_area": [155, 81, 434, 443],
        "landmarks": {
          "right_eye": [257.82974, 209.64787],
          "left_eye": [374.93427, 251.78687],
          "nose": [303.4773, 299.91144],
          "mouth_right": [228.37329, 338.73193],
          "mouth_left": [320.21982, 374.58798]
        }
  }
}

facial_area = faces[0]["facial_area"] 
landmarks = faces[0]["landmarks"]

cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)
#extract facial area 
img = cv2.imread(img_path) 
cv2.circle(img, tuple(landmarks["left_eye"]), 1, (0, 0, 255), -1) 
cv2.circle(img, tuple(landmarks["right_eye"]), 1, (0, 0, 255), -1) 
cv2.circle(img, tuple(landmarks["nose"]), 1, (0, 0, 255), -1) 
cv2.circle(img, tuple(landmarks["mouth_left"]), 1, (0, 0, 255), -1) 
cv2.circle(img, tuple(landmarks["mouthright"]), 1, (0, 0, 255), -1)
cv2.imwrite("sveta_retina.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(faces)

faces = RetinaFace.extract_faces(img_path = "img.jpg", align = True)
for face in faces:
  cv2.imshow(face)
