import numpy as np
from scipy import signal, ndimage

from models.hpe_models.hpe_model import HPE_Model

def angle_from_points(f_keypoints: list, kp_dict: dict, select_joints: list[str]) -> float:
    """
    Args:
        f_keypoints (list): list of skeleton keypoints for a single frame in format [x1, y1, c1, ... , xn, yn, cn]
        select_joints (list): list of three joint labels to select from keypoint dictionary
        kp_dict (dict): keypoint dictonary of the current model
    Returns:
        (float): calculated angle in radians
    """

    # Get keypoint indices
    joint_indices = [kp_dict.get(joint) for joint in select_joints]

    # Extract coordinates
    joint_d = np.array([f_keypoints[joint_indices[0]*3], f_keypoints[(joint_indices[0]*3)+1]])
    joint_e = np.array([f_keypoints[joint_indices[1]*3], f_keypoints[(joint_indices[1]*3)+1]])
    joint_f = np.array([f_keypoints[joint_indices[2]*3], f_keypoints[(joint_indices[2]*3)+1]])

    # Calculate vectors between points
    vector_A = joint_d - joint_e
    vector_B = joint_e - joint_f

    # Compute magnitudes
    mag = lambda v: np.sqrt(np.dot(v, v))
    mag_A, mag_B = mag(vector_A), mag(vector_B)

    # Prevent division by zero
    if mag_A == 0 or mag_B == 0:
        return np.nan

    # Calculate angle with clamping to avoid floating-point errors
    cos_theta = np.dot(vector_A, vector_B) / (mag_A * mag_B)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)

def calculate_angles(keypoints: list, model: HPE_Model) -> dict:
    
    # This dictionary defines the lables of joints needed for each important angle
    JOINT_CONNECTIONS: dict = {
        '0': ['left_foot_index', 'left_ankle', 'left_knee'],
        '1': ['right_foot_index', 'right_ankle', 'right_knee'],
        '2': ['left_ankle', 'left_knee', 'left_hip'],
        '3': ['right_ankle', 'right_knee', 'right_hip'],
        '4': ['left_knee', 'left_hip', 'right_hip'],
        '5': ['right_knee', 'right_hip', 'left_hip'],
        '6': ['left_hip', 'left_shoulder', 'left_elbow'],
        '7': ['right_hip', 'right_shoulder', 'right_elbow'],
        '8': ['left_shoulder', 'left_elbow', 'left_wrist'],
        '9': ['right_shoulder', 'right_elbow', 'right_wrist']
    }

    # Get keypoints dictionary
    kp_dict = model.get_kp_dict()

    # Initialise dictionary to store angles
    angles: dict = {}

    # Loop to calculate angles
    for theta, joints in JOINT_CONNECTIONS.items():

        # Ensure all joints exist in kp_dict
        if any(j not in kp_dict for j in joints):
            missing = [j for j in joints if j not in kp_dict]
            print(f"Skipping {theta=}: Missing keypoints {missing}")
            angles[theta] = None
            continue

        current_angle: list = []
        for f_keypoints in keypoints:
            current_angle.append(angle_from_points(f_keypoints, kp_dict, joints))
        angles[theta] = current_angle

    return angles

def derivative(input: list) -> list:

    derivative = []
    for i in range(1, len(input) - 1):
        derivative.append((input[i+1] - input[i-1]) * 15)

    return derivative


def gaussian_derivative(input: list, sigma: float = 4) -> list:

    filtered = ndimage.gaussian_filter(input, sigma)

    derived = derivative(filtered)
    
    return derived

def butterworth_derivative(input: list, order: int = 10, cutoff: float = 4) -> list:

    norm_cutoff: float = cutoff / (30 * 0.5)

    # Get the filter coefficients
    b, a = signal.butter(order, norm_cutoff, btype='low', analog=False)

    derived = derivative(signal.filtfilt(b, a, input))

    return derived
