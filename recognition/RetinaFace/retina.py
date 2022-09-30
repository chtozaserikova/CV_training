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

# применяет выравнивание к обнаруженным лицам с помощью своей функции извлечения
faces = RetinaFace.extract_faces(img_path = img_path, align = True)

# cv2.imshow("result", face) 
cv2.imwrite("sveta.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print(faces)