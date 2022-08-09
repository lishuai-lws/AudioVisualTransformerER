from torch import nn
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from config.config import shared_configs




class AudioWav2Vec2(nn.Module):
    def __init__(self,modelpath):
        super().__init__()
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained(modelpath,padding=True)
        self.model = Wav2Vec2Model.from_pretrained(modelpath)

    def forward(self, wavdata):
        data = self.tokenizer(wavdata, return_tensors="pt").input_values
        feature = self.model(data)
        return feature


class ResNet50(nn.Module):
    def __init__(self, modelpath):
        super.__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(modelpath)
        self.model = ResNetForImageClassification.from_pretrained(modelpath)

    def forward(self, image):
        inputs = self.feature_extractor(image, return_tensors="pt")
        feature = self.model(**inputs)
        return feature

def audio_Wav2Vec2(opts, wavdata):
    modelpath = opts.wav2vec2_base_960h
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(modelpath, padding=True)
    model = Wav2Vec2Model.from_pretrained(modelpath)
    input_values = tokenizer(wavdata, return_tensors="pt").input_values
    feature = model(input_values)
    return feature
def video_resnet50(opts, images):
    modelpath = opts.resnet50
    feature_extractor = AutoFeatureExtractor.from_pretrained(modelpath)
    model = ResNetForImageClassification.from_pretrained(modelpath)
    features = []
    for image in images:
        inputs = feature_extractor(image, return_tensors="pt")
        feature = model(**inputs)
        features = features.append(feature)

    return features
if __name__=="__main__":
    opts = shared_configs.get_data_embedding_args()
