import cv2
import os 
from ultralytics import YOLO
import cv2
from tqdm import tqdm
import subprocess
import librosa
import librosa.display

from audio_utils import merge_times, get_release_times, read_list_from_file, write_list_to_file, pair_bowler_faces_with_batting

def get_fps_ffmpeg(video_path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
           '-show_entries', 'stream=r_frame_rate', '-of', 
           'default=noprint_wrappers=1:nokey=1', video_path]
    
    output = subprocess.check_output(cmd).decode('utf-8').strip().split('/')
    fps = int(output[0]) / int(output[1])
    return int(fps)


def merge_clips(input_folder, output_video):
    # remove existing output video
    if os.path.exists(output_video):
        os.remove(output_video)

    clip_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".mp4")])

    with open("file_list.txt", "w") as f:
        for clip in clip_files:
            f.write(f"file '{clip}'\n")

    cmd = "ffmpeg -f concat -safe 0 -i file_list.txt -c copy " + output_video + " -loglevel quiet"
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

def extract_clip(start_time, clip_duration, video_path, output_path):
    # Seek to exact frame for more precise extraction
    #cmd = f'ffmpeg -ss {start_time:.3f} -i "{video_path}" -t {clip_duration} -c:v libx264 -preset fast "{output_path}" -loglevel quiet'
    cmd = f'ffmpeg -ss {start_time:.3f} -i "{video_path}" -t {clip_duration} -c:v libx264 -c:a aac -preset fast "{output_path}" -loglevel quiet'

    #print(f"Executing command: {cmd}")
    os.system(cmd)

def extract_clips(video_path, start_pairs, subtract_seconds_from_shot=25, clip_duration=10, output_folder="clips"):

    if not start_pairs:
        return

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


    with tqdm(total=len(start_pairs), desc="Processing frames") as pbar:
        #for i, frame_number in tqdm(enumerate(frames)):
        for i, (_, shot_sec) in tqdm(enumerate(start_pairs.items())):
            output_path = f"{output_folder}/clip_{i}.mp4"


            # subtract offset from bat shot
            start_time = shot_sec - subtract_seconds_from_shot
            start_time = max(0, start_time)
            
            extract_clip(start_time, clip_duration, video_path, output_path)
            pbar.update(1)

    #cap.release()
    print(f"Extracted clips saved in {output_folder}")

def get_start_pairs(video_path, frames):
    #peak_data_file = 'peak_data.txt'
    
    # check fps
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fps_ffmpeg = get_fps_ffmpeg(video_path)

    if fps != fps_ffmpeg:
        raise ValueError(f"FPS mismatch: {fps} != {fps_ffmpeg}")
    
    print(f"Video FPS: {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total frames: ', total_frames)

    # video frame to second dict
    frames_seconds = [(frame_number - 1) / fps for frame_number in frames]

    # save wav mono channel
    cmd = "ffmpeg -i input.mp4 -q:a 0 -map a output.wav -loglevel quiet"
    os.system(cmd)

    cmd = "ffmpeg -i output.wav -ac 1 output_mono.wav -loglevel quiet"
    os.system(cmd)

    # analyze audio
    y, sr = librosa.load('output_mono.wav', sr=None)
    release_times = merge_times(get_release_times(y, sr, percentile=99.50))
    #write_list_to_file(peak_data_file, release_times)
    #peaks = read_list_from_file(peak_data_file)
    print(release_times, 'release times')
    print(frames_seconds, 'frame secs')
    start_pairs = pair_bowler_faces_with_batting(release_times, frames_seconds)
    return start_pairs
    
if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    frames = list(map(int, sys.argv[2].strip('[]').split(',')))
    subtract_seconds_from_shot = int(sys.argv[3])
    clip_duration = int(sys.argv[4]) # 10
    out_video = sys.argv[5] # 10  
    output_folder = 'clips/'



    start_pairs = get_start_pairs(video_path, frames)
    print(start_pairs)
    extract_clips(video_path, 
                  start_pairs,
                  subtract_seconds_from_shot=subtract_seconds_from_shot, 
                  clip_duration=clip_duration, 
                  output_folder=output_folder
    )
    
    merge_clips(output_folder, out_video)

    # merge videos
    # ffmpeg -i p1.mp4 -i p2.mp4 -filter_complex "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]" -map "[outv]" -map "[outa]" output.mp4

