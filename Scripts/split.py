import os
from moviepy.editor import VideoFileClip

def split_videos(directory):

    output_directory = os.path.join(directory, "output")
    os.makedirs(output_directory, exist_ok=True)
    files = os.listdir(directory)

    for file in files:
        if file.endswith(".mp4") or file.endswith(".mp3"):
            file_path = os.path.join(directory, file)
            video = VideoFileClip(file_path)

            if video.duration > 120:#make it dynamic
                # Calculating the number of segments
                num_segments = int(video.duration / 120) + 1

                # Video split
                for i in range(num_segments):
                    start_time = i * 120
                    end_time = min((i + 1) * 120, video.duration)
                    segment = video.subclip(start_time, end_time)
                    output_file = os.path.join(output_directory, f"{file}_{i}.mp4")
                    segment.write_videofile(output_file)

                print(f"{file} has been split into {num_segments} segments.")
            else:
                print(f"{file} is already less than or equal to 2 minutes.")

if __name__ == "__main__":
    directory_path = input("Enter the directory path: ")
    split_videos(directory_path)
