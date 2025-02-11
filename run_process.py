import os
import sys
import tempfile
from download_video import download_video
from video_processor import extract_frames
from bowler_matcher import match_bowler
from clip_extractor import extract_clips, merge_clips

def run_pipeline(youtube_url, reference_image_path, output_video_path):
    try:
        # Create temporary directories for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            print("\n1. Downloading video...")
            video_dir = os.path.join(temp_dir, "video")
            os.makedirs(video_dir, exist_ok=True)
            download_video(youtube_url, video_dir)
            
            # Get the downloaded video file path (should be the only .mp4 file in the directory)
            video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            if not video_files:
                raise Exception("No video file was downloaded")
            video_path = os.path.join(video_dir, video_files[0])
            
            print("\n2. Extracting frames...")
            frames_dir = os.path.join(temp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            extract_frames(video_path, frames_dir)
            
            print("\n3. Matching bowler...")
            matched_frames = match_bowler(reference_image_path, frames_dir)
            if not matched_frames:
                raise Exception("No matching frames found for the bowler")
            print(f"Found {len(matched_frames)} matching frames")
            
            print("\n4. Extracting clips...")
            clips_dir = os.path.join(temp_dir, "clips")
            os.makedirs(clips_dir, exist_ok=True)
            
            # Default parameters for clip extraction
            subtract_seconds = 16
            clip_duration = 7
            skip_frames = 200
            
            extract_clips(
                video_path,
                matched_frames,
                subtract_seconds_from_start=subtract_seconds,
                clip_duration=clip_duration,
                output_folder=clips_dir
            )
            
            print("\n5. Merging clips...")
            merge_clips(clips_dir, output_video_path)
            
            print(f"\nPipeline completed successfully! Output video saved to: {output_video_path}")
            
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python run_process.py <youtube_url> <reference_image_path> <output_video_path>")
        sys.exit(1)
    
    youtube_url = sys.argv[1]
    reference_image_path = sys.argv[2]
    output_video_path = sys.argv[3]
    
    run_pipeline(youtube_url, reference_image_path, output_video_path)