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
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = self.linear(features.reshape(features.size(0), -1))
        # features = features.reshape(features.size(0), -1)
        # features = self.bn(self.linear(features))
        return features

'''
class CaptionGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
'''
class CaptionGenerator(nn.Module):
    def __init__(self, embedding_dim, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(CaptionGenerator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        captions = captions[:, :-1]
        embeddings = self.embed(captions)
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        # hiddens, _ = self.lstm(features)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        # packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        lstm_out, _ = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs
