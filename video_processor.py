import cv2
import os
import dlib
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from tqdm import tqdm


def contains_face(frame, detector, width=500):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height = int(frame.shape[0] * (width / frame.shape[1]))
    frame = cv2.resize(frame, (width, height))
    detected_faces = detector(frame)
    return len(detected_faces) > 0

def extract_frames(video_path, output_folder, frame_interval=1000):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total frames: ', total_frames)
    print("Extracting frames...")

    detector = dlib.get_frontal_face_detector()
    
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                if not contains_face(frame, detector):
                    continue
                frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
            
            frame_count += 1
            pbar.update(1)
 
    cap.release()
    print(f"Extracted frames saved in {output_folder}")

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    output_folder = sys.argv[2]
    extract_frames(video_path, output_folder)