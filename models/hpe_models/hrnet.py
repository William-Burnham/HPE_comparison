import numpy as np
import cv2 # type: ignore
import time

import mmcv # type: ignore
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

from models.hpe_models.hpe_model import HPE_Model

class HRNet_Model(HPE_Model):
    def __init__(self):
        HPE_Model.__init__(self)

        self.model_type = "HRNet"

        print("0.0")

        register_all_modules()

        config_file = 'configs\\td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
        print("0.1")
        checkpoint_file = 'checkpoints\\td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
        print("0.2")
        self.model = init_model(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
        print("0.3")


        self.KEYPOINT_DICT = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }

        self.CONNECTIONS = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[5,11],[6,8],[6,12],[7,9],[8,10],[11,12],[11,13],[12,14],[13,15],[14,16]]

    def predict(self,  file_path, conf=0):

        print("1")

        # Use MMCV for loading videos
        video = mmcv.VideoReader(file_path)

        keypoints = []

        print(len(video))

        # Loop through video frames
        for frame_i in range(len(video)):

            print("2")

            # Run inference on frame
            start = time.time()
            result = inference_topdown(self.model, video.read())
            end = time.time()

            print(f"{self.model_type} prediction for frame {frame_i} ({end-start} ms)")

            frame_data = []
            for keypoint_i in range(len(result[0].pred_instances.keypoints)):
                frame_data.append(result[0].pred_instances.keypoints[keypoint_i][0])
                frame_data.append(result[0].pred_instances.keypoints[keypoint_i][1])
                frame_data.append(result[0].pred_instances.keypoint_scores[keypoint_i])

            print("3")

            keypoints.append(frame_data)
        
        print("Done 1")

        return keypoints