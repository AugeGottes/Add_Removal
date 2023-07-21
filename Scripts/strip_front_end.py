from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

def remove_durations(input_file, output_file, start_duration, end_duration):
    ffmpeg_extract_subclip(input_file, start_duration, -end_duration, targetname=output_file)

input_file = "withads.mp4"
output_file = "outputwithads.mp4"
start_duration = 5 
end_duration = 0  

remove_durations(input_file, output_file, start_duration, end_duration)
