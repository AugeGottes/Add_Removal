# from Package
import moviepy.editor as mp
import os
import pickle
from pydub import AudioSegment
from headers import get_mfcc
import numpy as np


def delete(folder_path,section_to_delete):
    input_folder_path=folder_path[:-4]+".mp4"
    print(input_folder_path)
    
def merge(intervals):
    '''
    Merge intervals (will be there when we use sliding windows)
    '''
    v = [(interval[0], interval[1]) for interval in intervals]
    ans = []
        
    v.sort(key=lambda x: (x[0], x[1]))
        
    # for it in v:
    #     print(it[0], it[1])
        
    temp = (v[0][0], v[0][1])
    for it in v[1:]:
        if it[0] <= temp[1] and temp[1] <= it[1]:
            temp = (temp[0], it[1])
        elif it[0] <= temp[1] and it[1] <= temp[1]:
            pass
        else:
            ans.append(temp)
            temp = (it[0], it[1])
        
    ans.append(temp)
    return ans

def get_delete_Intervals(non_ad_predictions,largest_segment,segment_duration,total_duration_sec,folder_path):
    '''
    Gets the intervals to be deleted
    '''

    # sections_to_delete=[(segment_number*segment_duration-segment_duration,segment_duration*segment_number)
    #                     for segment_number in non_ad_predictions ]#formula lol

    sections_to_delete=[(segment_number*segment_duration-segment_duration,total_duration_sec if segment_number==largest_segment else
                         segment_duration*segment_number)
                        for segment_number in non_ad_predictions ]
    
    print(total_duration_sec)
    print(sections_to_delete)
    final_sections_to_delete=merge(sections_to_delete)
    print(final_sections_to_delete)
    print(folder_path)
    delete(folder_path,final_sections_to_delete)

def predict(audio_file_path,segment_duration,total_duration_sec,input_file_path):
    '''
    Predicts the label of the segments and stores them in a array also store the largest
    '''
    folder_path = '/home/debanjan/Desktop/Code/ML/CDSAML/src/segments/'  # Replace with the actual folder path containing the .mp3 files

    with open('svm_pickle', 'rb') as f:
        model = pickle.load(f)

    with open('scaler_pickle', 'rb') as f:
        scaler = pickle.load(f)

    
    ad_predictions = []
    non_ad_predictions = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp3'):
            file_path = os.path.join(folder_path, file_name)

            # Feature extraction
            audio_features = get_mfcc(file_name, path=folder_path)
            preprocessed_features = scaler.transform(audio_features.values.reshape(1, -1))

            #Classprediction
            predicted_label = model.predict(preprocessed_features)
            predicted_class = 0 if predicted_label[0] == 0 else 1

            # Storing the prediction in the corresponding array
            if predicted_class == 0:
                ad_predictions.append(int(file_name[:-4])) 
            else:
                non_ad_predictions.append(int(file_name[:-4]))  

    
    ad_predictions = np.array(ad_predictions)
    non_ad_predictions = np.array(non_ad_predictions)

    print("Ad Predictions:")
    print(ad_predictions)
    print("Non-Ad Predictions:")
    print(non_ad_predictions)


    largest_segment=np.max(np.concatenate([ad_predictions,non_ad_predictions]))
    print(largest_segment)
    get_delete_Intervals(non_ad_predictions,largest_segment,segment_duration,total_duration_sec,input_file_path)

def get_total_length(input_file_path):
    '''
    Get total length of the file
    '''
    audio = AudioSegment.from_file(input_file_path ,format='mp3')
    total_duration_ms = len(audio)
    total_duration_sec = total_duration_ms / 1000
    return total_duration_sec

def split(input_file_path, segment_duration):
    '''
    Split file into segments
    '''
    total_duration_sec=get_total_length(input_file_path)

    audio = AudioSegment.from_mp3(input_file_path)
    duration = len(audio)
    segment_duration_ms = segment_duration * 1000  

    segments = []
    for start_time in range(0, duration, segment_duration_ms):
        end_time = start_time + segment_duration_ms
        segment = audio[start_time:end_time]
        segments.append(segment)

    output_directory = "segments"
    os.makedirs(output_directory, exist_ok=True)

    for i, segment in enumerate(segments, start=1):
        output_file_path = os.path.join(output_directory, f"{i}.mp3")
        segment.export(output_file_path, format="mp3")

    sorted_files = sorted(os.listdir(output_directory), key=lambda f: int(os.path.splitext(f)[0]))

    
    print("Segmented files sorted by time:")
    for file_name in sorted_files:
        print(file_name)
    #no clue why i am segments
    predict(segments,segment_duration,total_duration_sec,input_file_path)
    
def display(directory):
    '''
    Receives a directory as inputs and prints all the files that end with mp4
    '''
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                print(file[:-4])
    
def convert_to_mp3(input_path,output_path):
    '''
    Convert mp4 to mp3
    '''
    video = mp.VideoFileClip(input_path)
    audio = video.audio
    audio.write_audiofile(output_path)
    video.close()


def main():
    print("Enter the file you want to be ad freed from the list of files")
    display(".")
    file_input=input("Enter your choice\n")
    convert_to_mp3(file_input+".mp4",file_input+".mp3")
    print(file_input)
    split(file_input+".mp3",20)#this can be user input


if __name__=="__main__":
    main()