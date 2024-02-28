import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

webcam = cv2.VideoCapture(0)

finger_tip_id = 8
finger_trail = []
last_activity_time = time.time()
trail_timeout = 10

while True:
    ret, frame = webcam.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        last_activity_time = time.time()

        for landmarks in results.multi_hand_landmarks:
            finger_tip = (int(landmarks.landmark[finger_tip_id].x * frame.shape[1]),
                          int(landmarks.landmark[finger_tip_id].y * frame.shape[0]))

            cv2.circle(frame, finger_tip, 10, (0, 255, 0), -1)

            finger_trail.append(finger_tip)
            for i in range(1, len(finger_trail)):
                cv2.line(frame, finger_trail[i-1], finger_trail[i], (0, 255, 0), 2)

            finger_trail = finger_trail[-50:]

    else:
        finger_trail = []

    cv2.imshow("Finger Tracking", frame)

    if time.time() - last_activity_time > trail_timeout:
        finger_trail = []

    if cv2.waitKey(1) & 0xFF == ord('f'):
        break

webcam.release()
cv2.destroyAllWindows()
