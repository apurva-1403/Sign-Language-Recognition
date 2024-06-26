import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Failed to open camera")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Mapping of labels to characters
labels_dict = {
    # 'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
    #            'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'nothing': 'nothing', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
    #            'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}

    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3',
    30: '4', 31: '5', 32: '6', 33: '7', 34: '8', 35: '9', }

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(frame_rgb)

    # Process hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract hand landmarks
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            data_aux = [(x - min(x_), y - min(y_)) for x, y in zip(x_, y_)]

            # Convert data_aux to a numpy array
            data_aux_array = np.array(data_aux)

            # Flatten the array if necessary
            data_aux_flat = data_aux_array.flatten()

            # Predict label
            prediction = model.predict([data_aux_flat])
            predicted_character = labels_dict[int(prediction[0])]

            # Calculate bounding box coordinates
            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Display frame
    cv2.imshow('frame', frame)

    # Check for exit key (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
