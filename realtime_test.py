import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model and labels
model = joblib.load("sign_model.pkl")
labels = np.load("labels.npy")

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,  # allow two hands
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

print("👉 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    prediction_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract landmarks for BOTH hands (pad if only one hand)
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

        # If only one hand detected, pad the other hand with zeros
        if len(results.multi_hand_landmarks) == 1:
            landmarks.extend([0.0, 0.0, 0.0] * 21)

        # If somehow more than two hands detected, truncate extra
        if len(landmarks) > 126:
            landmarks = landmarks[:126]

        # Convert to numpy
        landmarks = np.array(landmarks).reshape(1, -1)

        # Predict
        prediction = model.predict(landmarks)[0]
        sign_name = labels[prediction]
        prediction_text = f"Prediction: {sign_name}"

    # Display prediction on screen
    cv2.putText(frame, prediction_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

