import cv2 
from retinaface import RetinaFace

detector = RetinaFace(quality="normal")

rgb_image = detector.read("recognition\images\latest.jpg")
faces = detector.predict(rgb_image)
# x1 y1 x2 y2 left_eye right_eye nose left_lip right_lip
result_img = detector.draw(rgb_image,faces)

# save ([...,::-1] : rgb -> bgr )
# cv2.imwrite("data/result_img.jpg",result_img[...,::-1])

cv2.imshow("result",result_img[...,::-1])
cv2.waitKey()
