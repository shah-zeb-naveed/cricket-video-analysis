import cv2
import os 
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def merge_clips(input_folder, output_video):
    # remove existing output video
    if os.path.exists(output_video):
        os.remove(output_video)

    clip_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".mp4")])

    with open("file_list.txt", "w") as f:
        for clip in clip_files:
            f.write(f"file '{clip}'\n")

    cmd = "ffmpeg -f concat -safe 0 -i file_list.txt -c copy " + output_video
    os.system(cmd)
    os.remove("file_list.txt")




# Load YOLOv8 model trained on COCO (detects people, cricket bats, etc.)



def crop_image(img):    
    width = img.shape[1]
    crop_amount = int(width * 0.3)  # 20% of width
    cropped_img = img[:, crop_amount:width-crop_amount]
    return cropped_img

def detect_players(img):
    model = YOLO("yolov8n.pt")
    img = crop_image(img)
    results = model(img, verbose=False)  # Run inference
    num_players = len(results[0].boxes)
    #print('Results: ', num_players)
    return num_players

# develop a function that takes in a video frame number and then starts the clip from 
# that point in reverse and as soon as it detects number of players dropped by 1, it stops
# and that makrs the start of the clip. returns frame_number 
# using detect_players function to get number of players in the frame
def extract_clip_from_frame(video_path, frame_number):
    pass



def find_clip_start(video_path, start_frame):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Get initial player count
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read start frame.")
        cap.release()
        return None
    
    initial_players = detect_players(frame)

    # Go backward frame by frame
    for frame_number in range(start_frame - 1, 0, -1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break  # Stop if we can't read a frame

        player_count = detect_players(frame)

        # Stop when number of players drops by 1
        if player_count == initial_players - 1:
            cap.release()
            return frame_number, frame

    cap.release()
    return 0, 0  # If no change found, return start of video


def extract_clips(video_path, frames, subtract_seconds_from_start=25, clip_duration=10, output_folder="clips"):
    # Delete existing folder contents if folder exists
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error: {e}")
    
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video FPS: {fps}")
    
    # probe_cmd = f'ffprobe -i "{video_path}" -show_streams -select_streams a -loglevel error'
    # has_audio = os.system(probe_cmd) == 0
    # print(f"Video has audio: {has_audio}")

    with tqdm(total=len(frames), desc="Processing frames") as pbar:
        for i, frame_number in tqdm(enumerate(frames)):
            # If frame_3000.jpg was saved, it means it was the 3000th frame seen
            # So we need to use that exact frame number
            start_time = (frame_number - 1) / 25#fps
            #print(f"Frame number: {frame_number}")
            #print(f"Start time: {start_time//60} minutes {start_time%60} seconds")

            output_path = f"{output_folder}/clip_{i}.mp4"
            
            #print(f"Processing frame {frame_number} -> start time {start_time:.2f}s")
            

            # subtract 5 seconds from start time
            start_time -= subtract_seconds_from_start

            # Seek to exact frame for more precise extraction
            #cmd = f'ffmpeg -ss {start_time:.3f} -i "{video_path}" -t {clip_duration} -c:v libx264 -preset fast "{output_path}" -loglevel quiet'
            cmd = f'ffmpeg -ss {start_time:.3f} -i "{video_path}" -t {clip_duration} -c:v libx264 -c:a aac -preset fast "{output_path}" -loglevel quiet'

            #print(f"Executing command: {cmd}")
            os.system(cmd)
            pbar.update(1)
    cap.release()
    print(f"Extracted clips saved in {output_folder}")

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    frames = list(map(int, sys.argv[2].strip('[]').split(',')))
    subtract_seconds_from_start = int(sys.argv[3])
    clip_duration = int(sys.argv[4])
    skip_frames = int(sys.argv[5]) # 10
    out_video = sys.argv[6] # 10
    output_folder = 'clips/'


    #frames.sort()
    print('Frames: ', frames)
    # sort frame numbers and remove duplicates (duplicates means any frame that appears 
    # within 10 frames of each other) such that we retain the last occurrence

    filtered_frames = []
    i = len(frames) - 1
    while i >= 0:
        current = frames[i]
        # Add the current frame
        filtered_frames.append(current)
        # Skip all frames that are within 10 frames before the current one
        while i > 0 and current - frames[i-1] <= skip_frames:
            i -= 1
        i -= 1


    filtered_frames.reverse()
    print('Filtered frames: ', filtered_frames)

    extract_clips(video_path, 
                  filtered_frames, 
                  subtract_seconds_from_start=subtract_seconds_from_start, 
                  clip_duration=clip_duration, 
                  output_folder=output_folder
    )
    # python clip_extractor.py "Practice_#11_yasshi_sports_pt:1.mp4" "[18400, 20325, 20350, 22775, 22800, 27525, 29700, 29725, 32025, 32050, 35075, 37325, 37350, 40700, 40725, 43575, 43600, 43650, 46675, 49400, 53000, 53025, 55175, 55200, 57275, 59525, 62000, 62275, 67325, 67350, 71150, 71175, 71200, 77225, 77250]" 13 4 200 p1.mp4

    # part 1
    # [18400, 20325, 20350, 22775, 22800, 27525, 29700, 29725, 32025, 32050, 35075, 37325, 37350, 40700, 40725, 43575, 43600, 43650, 46675, 49400, 53000, 53025, 55175, 55200, 57275, 59525, 62000, 62275, 67325, 67350, 71150, 71175, 71200, 77225, 77250]

    # part 2
    # python clip_extractor.py "Practice_#11_Pt:2_yashi_sports.mp4" "[3650, 3675, 4550, 6125, 6150, 10750, 13875, 13900, 17550, 17575, 17600, 18600, 21400, 21425, 24625, 83125, 98250]" 13 4 200 p2.mp4


    merge_clips(output_folder, out_video)

    # merge videos
    # ffmpeg -i p1.mp4 -i p2.mp4 -c:v libx264 -crf 23 -preset fast -c:a aac -b:a 192k -shortest final.mp4

