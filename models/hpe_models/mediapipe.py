import numpy as np
import cv2 # type: ignore
import time

import mediapipe as mp
from mediapipe.tasks import python # type: ignore
from mediapipe.tasks.python import vision # type: ignore

from models.hpe_models.hpe_model import HPE_Model

# MediaPipe Pose: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models
class MediaPipe_Model(HPE_Model):
    def __init__(self):
        HPE_Model.__init__(self)

        self.model_type = 'mediapipe_pose'

        model_path = "./configs/mediapipe_pose_landmarker.task" # Ensure the pose landmarker is installed correctly to run this model
        self.base_options = python.BaseOptions(model_asset_path=model_path)

        self.KEYPOINT_DICT = {
            'nose': 0,
            'left_eye_(inner)': 1,
            'left_eye': 2,
            'left_eye_(outer)': 3,
            'right_eye_(inner)': 4,
            'right_eye': 5,
            'right_eye_(outer)': 6,
            'left_ear': 7,
            'right_ear': 8,
            'mouth_(left)': 9,
            'mouth_(right)': 10,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_pinky': 17,
            'right_pinky': 18,
            'left_index': 19,
            'right_index': 20,
            'left_thumb': 21,
            'right_thumb': 22,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32
        }

        self.CONNECTIONS = [[0,2],[0,5],[2,7],[5,8],[9,10],[11,12],[11,13],[11,23],[12,14],[12,24],[13,15],
                            [14,16],[15,17],[15,19],[15,21],[16,18],[16,20],[16,22],[17,19],[18,20],[23,24],
                            [23,25],[24,26],[25,27],[26,28],[27,29],[27,31],[28,30],[28,32],[29,31],[30,32]]

    def predict(self, file_path, conf=0):
        
        if conf: self.conf = conf

        # Video loader
        cap = cv2.VideoCapture(file_path)
        # Retrieve FPS from the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        keypoints = []

        options = vision.PoseLandmarkerOptions(
            base_options=self.base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )

        with vision.PoseLandmarker.create_from_options(options) as model:
            frame_i = 1
            ret = True
            while ret:
                ret, img = cap.read()
                # Loop handling
                if not ret: break

                # Calculate the frame's timestamp in milliseconds
                timestamp_ms = int(frame_i / fps * 1e3)
                # Convert image to a MediaPipe Image object
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
                # Run keypoint detection model on image
                start = time.time()
                result = model.detect_for_video(mp_img, timestamp_ms)
                end = time.time()
                print(f"{self.model_type} prediction for frame {frame_i} ({end-start} s)", end=' ')

                if not result.pose_landmarks:
                    frame_data = ['0'*3*len(self.KEYPOINT_DICT)]
                    print("No subject found")
                else:
                    print("")
                    # Extract desired data from result
                    frame_data = []
                    for landmark in result.pose_landmarks[0]:
                        frame_data.append(landmark.y)
                        frame_data.append(landmark.x)
                        frame_data.append(landmark.presence)
                    keypoints.append(frame_data)

                frame_i += 1

        return keypoints