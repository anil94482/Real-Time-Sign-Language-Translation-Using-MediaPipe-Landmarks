import cv2
import mediapipe as mp
import os
import numpy as np

# Ask for label
label = input("Enter the label name (e.g., hello, thanks, yes): ").strip().lower()

DATA_DIR = "data"
label_dir = os.path.join(DATA_DIR, label)
os.makedirs(label_dir, exist_ok=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,  # <-- allow both hands
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

sample_count = len(os.listdir(label_dir))
print(f"👉 Ready to capture samples for '{label}'")
print("👉 Press 's' to capture 30 samples at once")
print("👉 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"{label}: {sample_count} samples",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Sign Language Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print(f"⏳ Capturing 30 samples for '{label}' ...")
        burst_start = sample_count

        for i in range(30):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Prepare landmark vector for both hands
            all_landmarks = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        all_landmarks.extend([lm.x, lm.y, lm.z])

                # If only 1 hand, pad with zeros for the second
                if len(results.multi_hand_landmarks) == 1:
                    all_landmarks.extend([0.0, 0.0, 0.0] * 21)

            else:
                # No hand detected → all zeros (42 landmarks)
                all_landmarks = [0.0, 0.0, 0.0] * 42

            # Save
            npy_path = os.path.join(label_dir, f"{sample_count}.npy")
            np.save(npy_path, np.array(all_landmarks))
            sample_count += 1

            cv2.putText(frame, f"Capturing {i+1}/30", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Sign Language Data Collection", frame)
            cv2.waitKey(100)

        print(f"✅ Done! {sample_count - burst_start} samples captured.")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
