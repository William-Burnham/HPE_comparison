import matplotlib.pyplot as plt
import numpy as np
import cv2  # type: ignore
import os
from utils.data_utils import make_path

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    # Retrieve FPS from the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV format) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames, fps

def overlay_keypoints_on_video(video_path, output_dir, model_type, keypoints, connections, plot_confidence, confidence):
    figscale = 2.5

    video_frames, fps = load_video(video_path)
    filename = os.path.splitext(os.path.basename(video_path))[0]

    fig = plt.figure(figsize=(5*figscale, 5*figscale))

    # Remove boundary
    plt.axis('off')

    output_frames = []

    # Calculate width and height of video
    height, width, _ = video_frames[0].shape

    # Loop through each frame of keypoints
    for frame_index, frame_keypoints in enumerate(keypoints):
        # clear plot
        plt.clf()

        # Remove axes and adjust figure position
        ax = plt.gca()
        ax.set_axis_off()  # Hide axis
        ax.set_position([0, 0, 1, 1])  # Remove padding

        # Plot each joint
        for i in range(0, len(frame_keypoints), 3):
            if frame_keypoints[i+2] > confidence:
                x = int(frame_keypoints[i+1] * width)
                y = int(frame_keypoints[i] * height)
                if plot_confidence:
                    # Colour based on confidence
                    colour_conf = frame_keypoints[i+2]
                    # Bounding outliers between 0 and 1
                    if colour_conf > 1: colour_conf = 1
                    elif colour_conf < 0: colour_conf = 0
                    
                    plt.plot(x, y, '.', markersize=frame_keypoints[i+2]*7*figscale, color=(1-colour_conf,colour_conf,0))
                else:
                    plt.plot(x, y, '.', markersize=6*figscale, color=(0, 1, 0))

        # Plot each connection
        for connection in connections:
            if frame_keypoints[(connection[0]*3)+2] > confidence and frame_keypoints[(connection[1]*3)+2] > confidence:
                x1 = int(frame_keypoints[(connection[0]*3)+1] * width)
                y1 = int(frame_keypoints[(connection[0]*3)] * height)
                x2 = int(frame_keypoints[(connection[1]*3)+1] * width)
                y2 = int(frame_keypoints[(connection[1]*3)] * height)
                if plot_confidence:
                    # Colour based on confidence
                    connection_confidence = 0.5 * (frame_keypoints[(connection[0]*3)+2] + frame_keypoints[(connection[1]*3)+2]) # Mean joint confidence of the two connection joints
                    # Bounding outliers between 0 and 1
                    if connection_confidence > 1: connection_confidence = 1
                    elif connection_confidence < 0: connection_confidence = 0

                    plt.plot([x1, x2], [y1, y2], '-', linewidth=0.5*figscale, color=(1-connection_confidence, connection_confidence,0))
                else:
                    plt.plot([x1, x2], [y1, y2], '-', linewidth=0.5*figscale, color=(0, 0.8, 0))

        # Display the image with predicted joints
        image = video_frames[frame_index].astype(np.uint8)
        plt.imshow(image)
        
        fig.canvas.draw()  # Render the figure
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Convert RGB to BGR for OpenCV compatibility
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if frame_index == 0:
            # Create a video writer
            w_width, w_height, _ = img_bgr.shape
            output_path = make_path(output_dir, filename, model_type, "mp4")
            video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_width, w_height))

        # Write frame
        video_writer.write(img_bgr)

    video_writer.release()
    print(f"Video saved as {output_path}")