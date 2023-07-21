import os
import subprocess

def convert_to_mp3(input_file, output_file):
    try:
        subprocess.call(['ffmpeg', '-i', input_file, output_file])
        print(f"Conversion completed: {output_file}")
    except Exception as e:
        print(f"Error occurred while converting {input_file}: {str(e)}")

def convert_folder_to_mp3(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".mp4"):
            input_file = os.path.join(folder_path, file_name)
            output_file = os.path.join(folder_path, file_name[:-4] + ".mp3")
            convert_to_mp3(input_file, output_file)

folder_path = "C:/Users/Debanjan Das/Desktop/CDSAML/" 

convert_folder_to_mp3(folder_path)
