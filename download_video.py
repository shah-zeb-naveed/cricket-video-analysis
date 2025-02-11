from pytubefix import YouTube # This is solution
import sys

# little utility to parse out the resolution from a stream description like 1080p
def get_resolution(s):
    return int(s.resolution[:-1])



def download_video(url, save_path="."):
    res = 2000
    try:
        print("Fetching video details...")
        video_file = YouTube(url)
        
        print(f"Downloading: {video_file.title}")
        # Use progressive streams (contains both video and audio)
        # and filter for resolution <= 480p
        stream = video_file.streams.filter(only_video=True).order_by('bitrate').desc().first()
        print(f"Downloading: {stream.resolution}")
        stream.download(save_path)
        print("Download complete!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python youtube_downloader.py <YouTube_URL> [save_path]")
    else:
        video_url = sys.argv[1]
        save_directory = sys.argv[2] if len(sys.argv) > 2 else "."
        download_video(video_url, save_directory)
