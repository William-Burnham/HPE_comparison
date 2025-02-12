import os
import cv2
import torch
import numpy as np
import requests
import time

import mmcv
from mmcv import imread
#import mmengine
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

try:
    import mmdet
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
print(has_mmdet)

from mmdet.apis.inference import inference_detector, init_detector

from models.hpe_models.hpe_model import HPE_Model

class MMPose_Model(HPE_Model):
    def __init__(self, model_type="HRNet"):
        HPE_Model.__init__(self)

        self.model_type = model_type

        # Select paths for config and checkpoint
        if self.model_type.lower() == "hrnet":
            self.POSE_CONFIG_URL = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
            self.POSE_CONFIG_PATH = "configs/body_2d_keypoint/topdown_heatmap/coco/hrnet_w32_coco_256x192.py"
            self.POSE_CHECKPOINT = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"

            self.DET_CONFIG_URL = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
            self.DET_CONFIG_PATH = "configs/faster_rcnn/faster_rcnn_r50_fpn_coco.py"
            self.DET_CHECKPOINT = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
            
        else:
            raise Exception(f"(MMPose_Model): Unsupported model type {model_type}")

        # Ensure config files are downloaded
        self.download_file(self.POSE_CONFIG_URL, self.POSE_CONFIG_PATH)
        self.download_file(self.DET_CONFIG_URL, self.DET_CONFIG_PATH)
        
        # Ensure default runtime script is downloaded
        self.download_file(
            "https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/_base_/default_runtime.py",
            "configs/_base_/default_runtime.py"
        )

        # Load the selected model in two parts, detector and pose estimator
        self.detector, self.pose_estimator = self.load_model(self.POSE_CONFIG_PATH, self.POSE_CHECKPOINT, self.DET_CONFIG_PATH, self.DET_CHECKPOINT)
        
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

    def download_file(self, url, save_path):
        # Download file if it does not exist.
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if not os.path.exists(save_path):
            print(f"Downloading {url}...")
            response = requests.get(url, stream=True)
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {save_path}")
        else:
            print(f"{save_path} already exists, skipping download.")

    def load_model(self, pose_config, pose_checkpoint, det_config, det_checkpoint):
        # Build detector and pose estimator
        # MMPose tutorial: https://colab.research.google.com/drive/1rrCq6uPq6MhbNoslvBLqhljVKhqUO3RB#scrollTo=JjTt4LZAx_lK
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
        
        # build detector
        detector = init_detector(
            det_config,
            det_checkpoint,
            device=device
        )
        
        # build pose estimator
        pose_estimator = init_pose_estimator(
            pose_config,
            pose_checkpoint,
            device=device,
            cfg_options=cfg_options
        )

        return detector, pose_estimator

    def frame_inference(self, frame):
        # Run inference on single frame
        # MMPose tutorial: https://colab.research.google.com/drive/1rrCq6uPq6MhbNoslvBLqhljVKhqUO3RB#scrollTo=JjTt4LZAx_lK

        scope = self.detector.cfg.get('default_scope', 'mmdet')
        if scope is not None:
            init_default_scope(scope)

        start = time.time()
        detect_result = inference_detector(self.detector, frame)
        print(f"Inf det time = {time.time()-start}")

        pred_instance = detect_result.pred_instances.cpu().numpy()

        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                    pred_instance.scores > 0.3)]
        bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

        start = time.time()
        # predict keypoints
        pose_results = inference_topdown(self.pose_estimator, frame, bboxes)
        data_samples = merge_data_samples(pose_results)
        print(f"Pose inf time = {time.time()-start}")

        return data_samples

    def predict(self, file_path, conf=0):
        # Performs framewise pose estimation on video

        # Load video file
        cap = cv2.VideoCapture(file_path)
        keypoints_results = []

        # Get video dimensions for normalisation later
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        frame_i = 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for MMPose
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Begin timer
            start = time.time()

            # Run inference
            result = self.frame_inference(frame)

            # Display console message at each frame
            end = time.time()
            print(f"{self.model_type} prediction for frame {frame_i} ({end-start} s)")
            frame_i += 1

            # Extract keypoints & confidence scores
            if result is not None:
                # Take first prediction
                res = result.pred_instances[0]

                # Get keypoints and scores (normalised to the frame)
                keypoints = res.keypoints[0]
                normalised_kp = keypoints / [width, height]
                scores = res.keypoint_scores[0]
                
                # Stack scores with respective joint confidences then flatten the array
                kp_s = np.column_stack((normalised_kp, scores)).flatten()
                
                keypoints_results.append(kp_s)
            else:
                keypoints_results.append([0.0 for i in range(len(self.KEYPOINT_DICT)*3)])

        cap.release()
        return keypoints_results