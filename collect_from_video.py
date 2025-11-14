import cv2
import mediapipe as mp
import numpy as np
import os

# === USER SETTINGS ===
video_path = "goodbye.mp4"     # path to your downloaded video
label = "GoodBye"                     # name of the sign in the video
save_dir = os.path.join("data", label)
os.makedirs(save_dir, exist_ok=True)
# ======================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(video_path)
frame_count = len(os.listdir(save_dir))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Extract both hands (pad with zeros if only one hand)
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        if len(results.multi_hand_landmarks) == 1:
            landmarks.extend([0.0, 0.0, 0.0] * 21)

        np.save(os.path.join(save_dir, f"{frame_count}.npy"), np.array(landmarks))
        frame_count += 1

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Processing Video", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
