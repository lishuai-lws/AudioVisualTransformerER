from model.embedding import AudioWav2Vec2,ResNet50
from config.config import shared_configs
from utils.logger import LOGGER, add_log_to_file
from utils.misc import set_random_seed
from data.dataset import CMUMOSEIDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
opts = shared_configs.get_pretrain_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
LOGGER.info("device:{}".format(device))

set_random_seed(opts.seed)
LOGGER.info("Loading dataset...")
opts.get_data_process_args()
cmudataset = CMUMOSEIDataset(opts.cmumosei_audio_path,opts.cmumosei_video_path,opts.cmumosei_ids_path)
cmudataloader = DataLoader(dataset = cmudataset,batch_size=opts.batch_size,shuffle=True)
for audio, video in tqdm(cmudataloader):
    audiofeature = AudioWav2Vec2(audio)
    videofeature = []
    for image in video:
        feature = ResNet50(video)
        videofeature.append(feature)
    print(audiofeature)
    print(videofeature[0])