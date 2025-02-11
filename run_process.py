import os
import sys
import tempfile
from download_video import download_video
from video_processor import extract_frames
from bowler_matcher import match_bowler
from clip_extractor import extract_clips, merge_clips

def process_single_video(youtube_url, reference_image_path, temp_dir, part_num):
    print(f"\n=== Processing Video {part_num} ===")
    
    print("\n1. Downloading video...")
    video_dir = os.path.join(temp_dir, f"video_{part_num}")
    os.makedirs(video_dir, exist_ok=True)
    download_video(youtube_url, video_dir)
    
    # Get the downloaded video file path
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_files:
        raise Exception(f"No video file was downloaded for part {part_num}")
    video_path = os.path.join(video_dir, video_files[0])
    
    print("\n2. Extracting frames...")
    frames_dir = os.path.join(temp_dir, f"frames_{part_num}")
    os.makedirs(frames_dir, exist_ok=True)
    extract_frames(video_path, frames_dir)
    
    print("\n3. Matching bowler...")
    matched_frames = match_bowler(reference_image_path, frames_dir)
    if not matched_frames:
        raise Exception(f"No matching frames found for the bowler in part {part_num}")
    print(f"Found {len(matched_frames)} matching frames")
    
    print("\n4. Extracting clips...")
    clips_dir = os.path.join(temp_dir, f"clips_{part_num}")
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
    
    part_output = f"p{part_num}.mp4"
    print(f"\n5. Merging clips into {part_output}...")
    merge_clips(clips_dir, part_output)
    
    return part_output

def run_pipeline(youtube_urls, reference_image_path, output_video_path):
    try:
        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            part_videos = []
            
            # Process each video URL
            for i, url in enumerate(youtube_urls, 1):
                part_video = process_single_video(url, reference_image_path, temp_dir, i)
                part_videos.append(part_video)
            
            print("\nMerging all parts...")
            # Construct ffmpeg command for merging all parts
            inputs = " ".join(f"-i {v}" for v in part_videos)
            filter_parts = []
            for i in range(len(part_videos)):
                filter_parts.append(f"[{i}:v][{i}:a]")
            filter_complex = f"{' '.join(filter_parts)}concat=n={len(part_videos)}:v=1:a=1[outv][outa]"
            
            cmd = f'ffmpeg {inputs} -filter_complex "{filter_complex}" -map "[outv]" -map "[outa]" {output_video_path}'
            os.system(cmd)
            
            # Clean up part videos
            for video in part_videos:
                if os.path.exists(video):
                    os.remove(video)
            
            print(f"\nPipeline completed successfully! Final output video saved to: {output_video_path}")
            
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python run_process.py <reference_image_path> <output_video_path> <youtube_url1> [youtube_url2 ...]")
        sys.exit(1)
    
    reference_image_path = sys.argv[1]
    output_video_path = sys.argv[2]
    youtube_urls = sys.argv[3:]
    
    run_pipeline(youtube_urls, reference_image_path, output_video_path)