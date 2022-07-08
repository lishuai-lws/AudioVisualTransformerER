import os
import numpy as np
from moviepy.editor import *;


print("当前路径：",os.curdir)

#数据路径
data_path = '/home/lishuai/Dataset/CMU-MOSEI-RAW/Raw/'

processeddata_path = '/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/'
video_path = data_path + 'Videos/Segmented/Combined'
audio_path = processeddata_path +'audio/WAV_fromVideo'
# audio_path = data_path +'Audio/Full/COVAREP'
text_path = data_path +'Transcript/Segmented/Combined'
data_list = os.listdir(data_path)
if not os.path.exists(processeddata_path):
    os.mkdirs(processeddata_path)
print("data_list:",data_list)
def video():
    video_list = os.listdir(video_path)
    video_list.sort()
    print("video 1-5:",video_list[:5],"length:",len(video_list))

#从视频中提取音频数据，原始的音频数据不确定。
def audioFromVideo():
    process_video_data = "/home/lishuai/Dataset/CMU-MOSEI-RAW/processed_data/video/version_img_size_224_img_scale_1.3"
    video_list = os.listdir(process_video_data)
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    for video in video_list:
        print(f"precess:{video}")
        audio = VideoFileClip(video_path+'/'+video).audio
        audio.write_audiofile(audio_path+'/'+video[:-3]+'wav')


def audio():
    audio_list = []
    for file in os.listdir(audio_path):
        if file[-3:]=='wav':
            audio_list.append(file)
    audio_list.sort()
    print("audio 1-5:",audio_list[:5],"audio length:",len(audio_list))

# video()
audioFromVideo()
# audio()
