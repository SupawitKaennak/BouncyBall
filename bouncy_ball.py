import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Game parameters
screen_width, screen_height = 640, 480
object_position = [screen_width // 2, 100]  # Initial object position
object_radius = 30
gravity = 0.5  # Gravity acceleration
velocity_y = 0  # Initial downward velocity
ground_y = screen_height - object_radius
is_holding = False

# Capture video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame = cv2.resize(frame, (screen_width, screen_height))

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Apply gravity if not holding
    if not is_holding:
        velocity_y += gravity  # Increase velocity due to gravity
        object_position[1] += int(velocity_y)  # Update Y position

        # Collision with the ground
        if object_position[1] >= ground_y:
            object_position[1] = ground_y
            velocity_y = -velocity_y * 0.7  # Reverse and reduce velocity (bouncing)

            # Stop bouncing if velocity is too low
            if abs(velocity_y) < 2:
                velocity_y = 0

    # Draw object (ball)
    cv2.circle(frame, tuple(object_position), object_radius, (0, 255, 0), -1)

    # Hand tracking
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get positions of thumb and index fingertips
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

            # Calculate distance between thumb and index finger
            distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # If distance is small enough, consider it as "grabbing"
            if distance < 30:  # Threshold for grabbing
                is_holding = True
                velocity_y = 0  # Stop gravity while holding
                object_position = [index_x, index_y]  # Move object with index finger

            # If fingers are far apart, release the object
            else:
                is_holding = False

    # Display the frame
    cv2.imshow("Hand Tracking with Physics", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
