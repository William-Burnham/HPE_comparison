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
            'left eye (inner)': 1,
            'left eye': 2,
            'left eye (outer)': 3,
            'right eye (inner)': 4,
            'right eye': 5,
            'right eye (outer)': 6,
            'left ear': 7,
            'right ear': 8,
            'mouth (left)': 9,
            'mouth (right)': 10,
            'left shoulder': 11,
            'right shoulder': 12,
            'left elbow': 13,
            'right elbow': 14,
            'left wrist': 15,
            'right wrist': 16,
            'left pinky': 17,
            'right pinky': 18,
            'left index': 19,
            'right index': 20,
            'left thumb': 21,
            'right thumb': 22,
            'left hip': 23,
            'right hip': 24,
            'left knee': 25,
            'right knee': 26,
            'left ankle': 27,
            'right ankle': 28,
            'left heel': 29,
            'right heel': 30,
            'left foot index': 31,
            'right foot index': 32
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
                print(f"{self.model_type} prediction for frame {frame_i} ({end-start} ms)", end=' ')

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