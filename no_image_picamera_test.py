import argparse
import sys
import time

from picamera2 import Picamera2
from libcamera import controls
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

picam2 = Picamera2()
picam2.start_preview()
config = picam2.create_preview_configuration()
picam2.configure(config)
picam2.controls.ExposureTime = 30000
picam2.controls.AnalogueGain = 2.0
# picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
picam2.start()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

COUNTER, FPS = 0, 0
START_TIME = time.time()

row_size = 50  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 0)  # black
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

# Label box parameters
label_text_color = (255, 255, 255)  # white
label_font_size = 1
label_thickness = 2

recognition_frame = None
recognition_result_list = []

def save_result(result: vision.GestureRecognizerResult,
                  unused_output_image: mp.Image, timestamp_ms: int):
    global FPS, COUNTER, START_TIME

    # Calculate the FPS
    if COUNTER % fps_avg_frame_count == 0:
        FPS = fps_avg_frame_count / (time.time() - START_TIME)
        START_TIME = time.time()

    # recognition_result_list.append(result)
    COUNTER += 1
    
    if result.gestures:
        gesture = result.gestures[0]
        category_name = gesture[0].category_name
        print(f"\rGesture: {category_name}   |   FPS: {FPS}", end="", flush=True)

base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options,
                                        running_mode=vision.RunningMode.LIVE_STREAM,
                                        num_hands=1,
                                        min_hand_detection_confidence=0.5,
                                        min_hand_presence_confidence=0.5,
                                        min_tracking_confidence=0.5,
                                        result_callback=save_result)
recognizer = vision.GestureRecognizer.create_from_options(options)

try:
    while True:
        frame = picam2.capture_array()

        frame = cv2.flip(frame,1)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run gesture recognizer using the model.
        recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print("Error")
    print(e)
finally:
    picam2.stop()
    cv2.destroyAllWindows()