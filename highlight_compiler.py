import os

def merge_clips(input_folder, output_video="highlight.mp4"):
    clip_files = sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".mp4")])

    with open("file_list.txt", "w") as f:
        for clip in clip_files:
            f.write(f"file '{clip}'\n")

    cmd = "ffmpeg -f concat -safe 0 -i file_list.txt -c copy " + output_video
    os.system(cmd)
    os.remove("file_list.txt")

if __name__ == "__main__":
    import sys
    input_folder = sys.argv[1]
    merge_clips(input_folder)
