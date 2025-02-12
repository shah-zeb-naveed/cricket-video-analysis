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


def filter_frames(matched_frames, skip_frames):
    filtered_frames = []
    i = len(matched_frames) - 1
    while i >= 0:
        current = matched_frames[i]
        filtered_frames.append(current)
        while i > 0 and current - matched_frames[i-1] <= skip_frames:
            i -= 1
        i -= 1
    filtered_frames.reverse()
    print('Filtered frames: ', filtered_frames)
    return filtered_frames


def process_batch(batch, output_folder):
    detector = dlib.get_frontal_face_detector()
    #results = []
    for frame_number, frame in batch:
        if contains_face(frame, detector):
            #results.append((frame_number, frame))
            frame_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
    return 1

def extract_frames(video_path, output_folder, frame_interval=25):
    # remove all files in the output folder
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))

    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total frames: ', total_frames)
    print("Extracting frames...")

    # Create batches of frames
    batch_size = 32  # Adjust this based on your system's capabilities
    current_batch = []
    frame_count = 0
    
    # Number of processes to use (typically number of CPU cores)
    num_processes = multiprocessing.cpu_count()
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1
            
            if not ret:
                print('End of video')
                break
                
            if frame_count % frame_interval == 0:
                current_batch.append((frame_count, frame))
                
                # Process batch when it reaches batch_size
                if len(current_batch) >= batch_size:
                    # Submit batch for processing
                    future = executor.submit(process_batch, current_batch, output_folder)
                    #all_results.extend(future.result())
                    current_batch = []
        
        # Process remaining frames in the last batch
        if current_batch:
            future = executor.submit(process_batch, current_batch, output_folder)
            #all_results.extend(future.result())

    cap.release()

    # # Save the frames that contain faces
    # for frame_number, frame in all_results:
    #     frame_path = os.path.join(output_folder, f"frame_{frame_number}.jpg")
    #     cv2.imwrite(frame_path, frame)

    print(f"Extracted frames saved in {output_folder}")

    print('Filtering frames...')
    # read all frames in the output folder and return a list of frame numbers
    frames = []
    for file in os.listdir(output_folder):
        if file.endswith(".jpg"):
            frames.append(int(file.split("_")[1].split(".")[0]))
    
    frames = filter_frames(frames, skip_frames)
    frames = set(frames)

    # only retain these frames in the output folder
    for file in os.listdir(output_folder):
        if file.endswith(".jpg"):
            if int(file.split("_")[1].split(".")[0]) not in frames:
                os.remove(os.path.join(output_folder, file))

    return frames

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    output_folder = sys.argv[2]
    skip_frames = int(sys.argv[3])

    extract_frames(video_path, output_folder, skip_frames)