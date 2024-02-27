import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
    frameread, frame = webcam.read()

    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    coordinates = trained_face_data.detectMultiScale(grayscaled, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("FLX FACE DETECTOR", frame)

    if cv2.waitKey(1) & 0xFF == ord('f'):
        break

webcam.release()
cv2.destroyAllWindows()
