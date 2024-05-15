from asyncio import Queue
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from random import random
from threading import Lock, Thread
from datetime import datetime, time
import mediapipe as mp
import cv2
import numpy as np
import csv
import copy
import argparse
import itertools
from collections import deque
from model import GestureClassifier

"""
Background Thread
"""
thread = None
thread_lock = Lock()
data_queue = []
coordinates = []
labels=[]


app = Flask(__name__)
app.config['SECRET_KEY'] = 'donsky!'
socketio = SocketIO(app, cors_allowed_origins='*')

"""
Get current date time
"""
def get_current_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y %H:%M:%S")
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


"""
Generate random sequence of dummy sensor values and send it to our clients
"""
def background_thread():
    print("Generating random sensor values")
    global coordinates
    global labels
    while True:
        if coordinates:
            data = coordinates.pop(0)
            label = float(labels.pop(0))
            dummy_sensor_value = label
            print("Processing data:", label)
        else:
            dummy_sensor_value = 1.0
        socketio.emit('updateFingersPositions', {"x1": data[0], "y1": data[1], "x2": data[2], "y2": data[3]})
        socketio.sleep(0.3)

def get_frame():
    global data_queue
    cap = cv2.VideoCapture(0) 
    while True:
        ret, frame = cap.read() # ret - return value
            # We change the color format here from BGR to RGB to make the model work
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Flip image on horizontal
        image = cv2.flip(image, 1)
            # Set flag
        image.flags.writeable = False
        data_queue.append(frame)
        socketio.sleep(0.2) # Adjust the sleep time as needed

def image_proccessing():
    print("Thread entered")
    global data_queue
    global coordinates
    global labels
    mp_hands = mp.solutions.hands # hands models
    gesture_classifier = GestureClassifier()
    hands = mp_hands.Hands(
                       min_detection_confidence=0.8,
                       min_tracking_confidence=0.5,
                       max_num_hands=1)
    print("Variables were set")
    while True:
        if data_queue:
            image = data_queue.pop(0)
            results = hands.process(image)
            if results.multi_hand_landmarks:
                for  hand in results.multi_hand_landmarks:
                    middle_finger_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    index_finger_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                    x1 = index_finger_tip.x
                    y1 = index_finger_tip.y
                    x2 = middle_finger_tip.x
                    y2 = middle_finger_tip.y
                    coordinates.append((x1, y1, x2, y2))


                    landmark_list = calc_landmark_list(image, hand)
                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(
                        landmark_list)
                    hand_sign_id = gesture_classifier(pre_processed_landmark_list)
                    labels.append(hand_sign_id)
        else:
            print("No elements in the queue were found")
        socketio.sleep(0.2)


"""
Serve root index file
"""
@app.route('/')
def index():
    return render_template('index.html')

"""
Decorator for connect
"""
@socketio.on('connect')
def connect():
    global thread
    print('Client connected')

    global thread
    with thread_lock:
        if thread is None:
            thread = Thread(target=get_frame)
            thread.daemon = True
            thread.start()
            socketio.start_background_task(background_thread)
            socketio.start_background_task(image_proccessing) 

"""
Decorator for disconnect
"""
@socketio.on('disconnect')
def disconnect():
    print('Client disconnected',  request.sid)

@app.route('/add_hello', methods=['POST'])
def add_hello():
    data = request.get_json()
    if 'string' in data:
        responce_string = data['string'] + ", hello!"
        return {"responce": responce_string}
    else:
        return {"error": "string not provided"}, 400

if __name__ == '__main__':
    socketio.run(app);