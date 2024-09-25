import cv2
import numpy as np
import mediapipe as mp
import random

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

score = 0
best_score = 0
falling_objects = []
object_radius = 40
fall_speed = 10

def create_falling_object():
    x = random.randint(object_radius, width - object_radius)
    y = 0
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return {'position': [x, y], 'color': color}

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:  
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if len(falling_objects) < 1:
            falling_objects.append(create_falling_object())

        for obj in falling_objects:
            obj['position'][1] += fall_speed
            cv2.circle(frame, (obj['position'][0], obj['position'][1]), object_radius, obj['color'], -1)

            if obj['position'][1] > height:
                score = 0
                falling_objects.remove(obj)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
               
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4)  
                )

                tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
                tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)

                for obj in falling_objects:
                    distance = np.sqrt((tip_x - obj['position'][0]) ** 2 + (tip_y - obj['position'][1]) ** 2)
                    if distance < object_radius:
                        score += 1
                        falling_objects.remove(obj)
                        break

        if score > best_score:
            best_score = score

        pink_color = (203, 192, 255) 
        cv2.putText(frame, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, pink_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f'High Score: {best_score}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, pink_color, 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking Game', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
