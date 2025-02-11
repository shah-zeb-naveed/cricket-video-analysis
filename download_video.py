from pytubefix import YouTube # This is solution
import sys
#import ffmpeg
import os 


# little utility to parse out the resolution from a stream description like 1080p
def get_resolution(s):
    return int(s.resolution[:-1])


def download_video(url, save_path="."):
    try:
        print("Fetching video details...")
        yt = YouTube(url)

        print(f"Downloading: {yt.title}")

        # Get highest-resolution video (video-only stream)
        video_stream = yt.streams.filter(adaptive=True, file_extension="mp4")\
                                 .order_by("resolution")\
                                 .desc()\
                                 .first()

        # Get highest-quality audio (audio-only stream)
        audio_stream = yt.streams.filter(only_audio=True, file_extension="mp4")\
                                 .order_by("abr")\
                                 .desc()\
                                 .first()

        if not video_stream or not audio_stream:
            print("Error: Could not find suitable video or audio streams.")
            return

        # Define output file paths
        video_filename = os.path.join(save_path, "video.mp4")
        audio_filename = os.path.join(save_path, "audio.mp4")
        output_filename = os.path.join(save_path, f"{yt.title}.mp4").replace(" ", "_")  # Remove spaces to avoid issues

        print(f"Downloading video ({video_stream.resolution})...")
        video_stream.download(save_path, filename="video.mp4")

        print(f"Downloading audio ({audio_stream.abr})...")
        audio_stream.download(save_path, filename="audio.mp4")

        print("Merging video and audio using FFmpeg...")

        # Correctly use ffmpeg with file paths
        cmd = f'ffmpeg -i "{video_filename}" -i "{audio_filename}" -c:v copy -c:a aac "{output_filename}"'
        os.system(cmd)

        print(f"Download complete! Merged file saved as: {output_filename}")

        # Clean up temporary files
        os.remove(video_filename)
        os.remove(audio_filename)

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python youtube_downloader.py <YouTube_URL> [save_path]")
    else:
        video_url = sys.argv[1]
        save_directory = sys.argv[2] if len(sys.argv) > 2 else "."
        download_video(video_url, save_directory)
