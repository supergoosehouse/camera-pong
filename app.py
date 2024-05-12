import mediapipe as mp
import cv2
import numpy as np
import csv
import copy
import argparse
import itertools
from collections import deque
from model import GestureClassifier

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_static_image_mode', action='store_true')
    args = parser.parse_args()
    use_static_image_mode = args.use_static_image_mode
    mp_hands = mp.solutions.hands # hands models
    with open('python-flask-socketio\model\gesture_classifier_label.csv',
                encoding='utf-8-sig') as f:
            gesture_classifier_labels = csv.reader(f)
            gesture_classifier_labels = [
                row[0] for row in gesture_classifier_labels
            ]
    # joint of the hand(red dots) are called landmarks

    # Web cam feed, on which we arfe going to overlap medipipe model
    cap = cv2.VideoCapture(1) # the number is the divice's camera
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
            # chacks if there are any coordinates in the list
            if results.multi_hand_landmarks:
                for  hand, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)


                    hand_sign_id = gesture_classifier(pre_processed_landmark_list)

                    index_finger_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                    x = index_finger_tip.x
                    y = index_finger_tip.y
                    print(hand_sign_id)

                    # Optional: Convert normalized coordinates to pixel values
                    # (if you need absolute coordinates within the image)
                    normalized_x = int(index_finger_tip.x * image.shape[1])
                    normalized_y = int(index_finger_tip.y * image.shape[0])

    # closing everything
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()