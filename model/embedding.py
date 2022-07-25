from torch import nn
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
from transformers import AutoFeatureExtractor, ResNetForImageClassification




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
