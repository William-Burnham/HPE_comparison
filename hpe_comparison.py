import argparse
import json

from utils.data_utils import load_file_paths, save_keypoints, reposition_keypoints
from utils.plot_utils import overlay_keypoints_on_video

import traceback

def main(config):
    #results = {}

    # Loads the paths of all files within 'input_folder' into a list
    file_paths = load_file_paths(config["input_folder"])

    # Run selected models on all files within 'file_paths':
    # --------------------------------------------------- #
    # HRNet
    if config["hrnet"]:
        # Load hrnet model
        from models.hpe_models.mmpose_model import MMPose_Model
        model = MMPose_Model(model_type="HRNet")

        # Predict on all files in input folder
        for file_path in file_paths:
            keypoints = model.predict(file_path=file_path, conf=config["confidence"])
            
            # Keypoints x and y need to be swapped to be consistent with other models
            keypoints = reposition_keypoints(
                keypoints,
                swap_xy = True
            )

            save_output(model, keypoints, file_path, config)

    # MediaPipe Pose
    if config["mediapipe_pose"]:
        # Load mediapipe model
        from models.hpe_models.mediapipe import MediaPipe_Model
        model = MediaPipe_Model()

        # Predict on all files in input folder
        for file_path in file_paths:
            keypoints = model.predict(file_path=file_path, conf=config["confidence"])
            #results[str(model.get_model_type())] = keypoints
            save_output(model, keypoints, file_path, config)

    # MoveNet Lightning
    if config["movenet_lightning"]:
        # Load movenet model
        from models.hpe_models.movenet import MoveNet_Model
        model = MoveNet_Model(model_type="movenet_lightning")

        # Predict on all files in input folder
        for file_path in file_paths:
            keypoints = model.predict(file_path=file_path, conf=config["confidence"])
            #results[str(model.get_model_type())] = keypoints
            save_output(model, keypoints, file_path, config)
    
    # MoveNet Thunder
    if config["movenet_thunder"]:
        # Load movenet model
        from models.hpe_models.movenet import MoveNet_Model
        model = MoveNet_Model(model_type="movenet_thunder")

        # Predict on all files in input folder
        for file_path in file_paths:
            keypoints = model.predict(file_path=file_path, conf=config["confidence"])
            #results[str(model.get_model_type())] = keypoints
            save_output(model, keypoints, file_path, config)

    # YoloPose (Ultralytics)
    if config["yolopose"]:
        # Load yolopose model
        from models.hpe_models.yolopose import Yolo_Model
        model = Yolo_Model()

        # Predict on all files in input folder
        for file_path in file_paths:
            keypoints = model.predict(file_path=file_path, conf=config["confidence"])
            
            # Keypoints x and y need to be swapped to be consistent with other models
            keypoints = reposition_keypoints(
                keypoints,
                swap_xy = True
            )
            #results[str(model.get_model_type())] = keypoints
            save_output(model, keypoints, file_path, config)

def save_output(model, keypoints, file_path, config):
    if config["do_save_keypoints"]: save_keypoints(
        model_type=model.get_model_type(),
        keypoint_dict=model.get_kpd(),
        keypoints=keypoints,
        save_dir=config["keypoints_output_folder"],
        file_path = file_path
    )
        
    if config["do_save_overlay"]: overlay_keypoints_on_video(
        video_path=file_path,
        output_dir=config["overlay_output_folder"],
        model_type=model.get_model_type(),
        keypoints=keypoints,
        connections=model.get_connections(),
        plot_confidence=config["do_plot_confidence"],
        confidence=model.get_confidence()
    )

if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser(description='HPE_Comparative_Study')

        parser.add_argument('--config_json', '-config', default='configuration.json', type=str)

        args = parser.parse_args()

        config_file = args.config_json
        with open(config_file) as json_file:
            config = json.load(json_file)

        main(config)

    except Exception as e:
        
        print("Caught an exception:")
        traceback.print_exc()