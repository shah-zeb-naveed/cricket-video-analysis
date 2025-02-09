import cv2
import os 

def extract_clips(video_path, timestamps, clip_duration=5, output_folder="clips", frame_interval=1000):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video FPS: {fps}")
    
    for i, frame_number in enumerate(timestamps):
        # If frame_3000.jpg was saved, it means it was the 3000th frame seen
        # So we need to use that exact frame number
        start_time = (frame_number) / fps
        output_path = f"{output_folder}/clip_{i}.mp4"
        
        print(f"Processing frame {frame_number} -> start time {start_time:.2f}s")
        
        # Seek to exact frame for more precise extraction
        cmd = f'ffmpeg -ss {start_time:.3f} -i "{video_path}" -t {clip_duration} -c:v libx264 -preset fast "{output_path}"'
        print(f"Executing command: {cmd}")
        os.system(cmd)

    cap.release()
    print(f"Extracted clips saved in {output_folder}")

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    timestamps = list(map(int, sys.argv[2].strip('[]').split(',')))
    extract_clips(video_path, timestamps)
    # python clip_extractor.py input_video.mp4 "[100, 450, 900]"

