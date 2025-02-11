import cv2
import os
import dlib
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

from clip_extractor import detect_players


def preprocess_frame(frame, width=2500):
    #print('Channels: ', len(frame.shape))
    if len(frame.shape) == 3:  # If image is color (has 3 channels)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height = int(frame.shape[0] * (width / frame.shape[1]))
    frame = cv2.resize(frame, (width, height))
    return frame

def contains_face(frame, detector, width=2500):
    frame = preprocess_frame(frame, width)
    #detected_faces = detector(frame)
    detected_faces, scores, idx = detector.run(frame, adjust_threshold=0.1)
    # for _, score in zip(detected_faces, scores):
    #     print(f"Face detected with confidence: {score}")
        
    return len(detected_faces) > 0

def extract_frames(video_path, output_folder, frame_interval=25):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #print(f"Video FPS: {fps}")
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total frames: ', total_frames)
    print("Extracting frames...")

    detector = dlib.get_frontal_face_detector()
    
#    with tqdm(total=total_frames, desc="Processing frames") as pbar:
    prev_frame = None
    cur_start_frame_number = None
    cur_start_frame = None


    while cap.isOpened():
        ret, frame = cap.read()
        
        frame_count += 1
        #pbar.update(1)

        if not ret:
            print('End of video')
            break

        prev_frame = frame
        
        if frame_count % frame_interval == 0:
            num_players = detect_players(frame)

            
            if not contains_face(frame, detector):
                if num_players > 1 and cur_start_frame_number is None:
                    print('Start of clip:', frame_count)
                    cur_start_frame_number = frame_count
                    cur_start_frame = frame
                continue

            print('Found face in frame: ', frame_count)
            
            if cur_start_frame is not None:
                frame_path = os.path.join(output_folder, f"frame_{cur_start_frame_number}_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)

                frame_path = os.path.join(output_folder, f"frame_{cur_start_frame_number}.jpg")
                cv2.imwrite(frame_path, cur_start_frame)

            cur_start_frame_number = None
            cur_start_frame = None
# process last frame
    try:
        if contains_face(prev_frame, detector):
            frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
    except Exception as e:
        print(f"Error processing last frame: {e}")

    cap.release()
    print(f"Extracted frames saved in {output_folder}")

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    output_folder = sys.argv[2]

    # remove all files in the output folder
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))

    extract_frames(video_path, output_folder)