from facial_emotion_recognition import EmotionRecognition
import cv2

model = EmotionRecognition('cpu')

cam = cv2.VideoCapture(0)

while True:
    img = cam.read()[1]

    face = model.recognise_emotion(img,return_type='bgr')

    cv2.imshow("",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cam.release()
