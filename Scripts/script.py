import os
import subprocess
from pytube import YouTube

def download_video(url, output_path):
    try:
        youtube = YouTube(url)
        video = youtube.streams.get_highest_resolution()
        video.download(output_path)

        # Video->mp3
        video_path = output_path + video.default_filename
        mp3_path = output_path + video.default_filename[:-4] + ".mp3"
        subprocess.call(['ffmpeg', '-i', video_path, mp3_path])

        os.remove(video_path)

        print("Video downloaded and converted to MP3 successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  
output_directory = "C:/Users/Debanjan Das/Desktop/CDSAML"  

download_video(video_url, output_directory)
