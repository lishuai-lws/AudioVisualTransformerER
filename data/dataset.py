import pandas as pd
from torch.utils.data import Dataset
from config.config import shared_configs
import  os
import librosa
import torchvision.transforms as transformer
from PIL import Image
import numpy as np

class CMUMOSEIDataset(Dataset):
    def __init__(self,audio_path,video_path,ids_path):
        ids = np.array(pd.read_csv(ids_path))
        ids = ids.reshape(ids.shape[0],).tolist()
        print(ids[:5])
        self.videoTransformer = transformer.Compose([
            transformer.CenterCrop([256,256]),
            transformer.ToTensor()
        ])
        wavedatas = []
        videodatas = []
        for id in ids[:5]:
            print("id:",id)
            wave_data, samplerate = librosa.load(os.path.join(audio_path, id+".wav"))
            wavedatas.append(wave_data)
            videodir = os.path.join(video_path,id+"_aligned")
            imglist = os.listdir(videodir)
            video = []
            for image in imglist:
                imgpath = os.path.join(videodir,image)
                img = Image.open(imgpath)
                video.append(np.array(img.getdata()))
                img.close()
            videodatas.append(video)
        self.audiodata = wavedatas
        self.videodata = videodatas
        self.ids = ids[:5]


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        return self.audiodata[item], self.videodata[item]
