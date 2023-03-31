import cv2
import mediapipe as mp
import time 


class FaceDetector():
    def __init__(self, minDetectionConfidence = 0.5):

        self.minDetectionConfidence = minDetectionConfidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfidence)

    def findFaces(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bb_white = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bb_pink = int(bb_white.xmin*iw), int(bb_white.ymin*ih), int(bb_white.width*iw), int(bb_white.height*ih)
                bboxs.append([id, bb_pink, detection.score])
                cv2.rectangle(img, bb_pink, (255, 0, 255), 2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bb_pink[0], bb_pink[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return img, bboxs



def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)
        print(bboxs)
    
        cv2.imshow('image', img)
        if cv2.waitKey(1) == 27:
            break
    

if __name__ == '__main__':
    main()