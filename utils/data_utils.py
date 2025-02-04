import os
import csv

def load_file_paths(directory):
    file_paths = os.listdir(directory)
    for i in range(len(file_paths)): file_paths[i] = os.path.join(directory, file_paths[i])
    return file_paths

def make_path(save_dir, file_path, model_type, extension):
    split_filename = os.path.splitext(os.path.basename(file_path))[0] # Get the filename without extension from file_path
    return f"{os.path.join(save_dir, split_filename)}-{model_type}.{extension}" # File name formatted as "original_filename"-"model_type" e.g. "video-yolov8n-pose.csv"

def save_keypoints(model_type, keypoint_dict, keypoints, save_dir, file_path):
    save_path = make_path(save_dir, file_path, model_type, "csv")
    
    # Build labels for first row
    first_row = []
    for key, value in keypoint_dict.items():
        first_row.append(f"X{value}-{key}")
        first_row.append(f"Y{value}-{key}")
        first_row.append(f"C{value}-{key}")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(first_row)
        writer.writerows(keypoints)

def reposition_keypoints(keypoints, swap_xy=False, flip_x=False, flip_y=False, confidence_factor=1, deconf_zeros=True):
    """
    keypoints: list of keypoints for each frame
    swap_xy: swap the x and y values of each keypoint
    flip_x: flip x positions over the centre
    flip_y: flip y positions over the centre
    confidence_factor: factor to multiply all confidence levels by
    deconf_zeros: for all positions predicted at (0.0, 0.0), set confidence to 0.0
    """
    
    # Rearrange points to correct placements
    for keypoint_i, keypoint_value in enumerate(keypoints):

        for joint_i in range(0, len(keypoint_value), 3):

            if swap_xy:
                temp = keypoint_value[joint_i]
                keypoint_value[joint_i] = keypoint_value[joint_i+1]
                keypoint_value[joint_i+1] = temp
            
            if flip_x:
                keypoint_value[joint_i] = 1 - keypoint_value[joint_i]

            if flip_y:
                keypoint_value[joint_i+1] = 1 - keypoint_value[joint_i+1]

            if deconf_zeros and keypoint_value[joint_i] == 0.0 and keypoint_value[joint_i+1] == 0.0:
                keypoint_value[joint_i+2] = 0.0

            keypoint_value[joint_i+2] *= confidence_factor
        
        keypoints[keypoint_i] = keypoint_value

    return keypoints