import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from model import GestureClassifier
from flask import Flask, render_template, request
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'donsky!'
socketio = SocketIO(app, cors_allowed_origins='*')

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list




def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/gesture.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def main():
    mode=0
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_static_image_mode', action='store_true')
    args = parser.parse_args()
    use_static_image_mode = args.use_static_image_mode
    use_brect = True
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands # hands models
    with open('model/gesture_classifier_label.csv',
                encoding='utf-8-sig') as f:
            gesture_classifier_labels = csv.reader(f)
            gesture_classifier_labels = [
                row[0] for row in gesture_classifier_labels
            ]
    # joint of the hand(red dots) are called landmarks

    # Web cam feed, on which we arfe going to overlap medipipe model
    cap = cv2.VideoCapture(0) # the number is the divice's camera
    # instantiation mediapipe hands model
    # min_tracking_confidence 
    # min_detection_confidence - how accurate out model will be
    # 100% will most likely overfit
    with mp_hands.Hands(static_image_mode=use_static_image_mode, min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        gesture_classifier = GestureClassifier()
        finger_gesture_history = deque(maxlen=16)
        while cap.isOpened():
            key = cv2.waitKey(10)
            if key == 27:  # ESC
                break
            number, mode = select_mode(key, mode)

            ret, frame = cap.read() # ret - return value
            # We change the color format here from BGR to RGB to make the model work
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Flip image on horizontal
            image = cv2.flip(image, 1)
            debug_image = copy.deepcopy(image)
            # Set flag
            image.flags.writeable = False
            # Pass image to model
            results = hands.process(image)
            # Set flag to true
            image.flags.writeable = True
            # Convert image format back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            debug_image = copy.deepcopy(image)
            socketio.emit("frame", {"image": debug_image.tobytes()})
            # chacks if there are any coordinates in the list
            if results.multi_hand_landmarks:
                for  hand, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):


                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand)
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)

                    # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list)

                    hand_sign_id = gesture_classifier(pre_processed_landmark_list)


                    # Drawing part
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    mp_drawing.draw_landmarks(debug_image, hand, mp_hands.HAND_CONNECTIONS)
                    debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    str("X: " + str(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x) + "\nY: " + str(hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y))
                    #gesture_classifier_labels[hand_sign_id],
                    )


                    middle_finger_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    index_finger_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    x1 = index_finger_tip.x
                    y1 = index_finger_tip.y
                    x2 = middle_finger_tip.x
                    y2 = middle_finger_tip.y

                    socketio.emit('updateFingersPositions', {"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                    # Optional: Convert normalized coordinates to pixel values
                    # (if you need absolute coordinates within the image)
                    normalized_x = int(index_finger_tip.x * image.shape[1])
                    normalized_y = int(index_finger_tip.y * image.shape[0])



            cv2.imshow('Hand Tracking', debug_image)

    # closing everything
        
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def connect():
    socketio.start_background_task(main)

if __name__ == '__main__':
    socketio.run(app);
    #main();