import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

success = cam.isOpened()
if success == False:
    print('Error: Camera could not be opened')
else:
    print('Success: Grabbed the camera') 

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} записан!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()