import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class FeatureExtractor(nn.Module):
    def __init__(self, embedding_dim):
        """Load the pretrained ResNet-34 and replace top fc layer."""
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet34(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = self.linear(features.reshape(features.size(0), -1))
        # features = features.reshape(features.size(0), -1)
        features = self.bn(features)
        return features

class CaptionGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        """Decode image feature vectors and generates captions."""
        captions = captions[:, :-1]
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, features, states=None, max_len=20):
        """Generate captions for an image"""
        caption = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            caption.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        caption = torch.stack(caption, 1)
        return caption
