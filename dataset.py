import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from vocabulary import create_vocabulary
from PIL import Image
import nltk
nltk.download('punkt')


class CocoDataset(Dataset):
    """COCO dataset."""

    def __init__(self, json_path, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.coco = COCO(json_path)
        self.vocabulary, self.ids = create_vocabulary(self.coco)
        self.word_to_ix = {}
        self.id_to_word = {}
        self.root_dir = root_dir
        self.transform = transform
        self.map_vocab_to_number()

    def map_vocab_to_number(self):
        value = 0
        for word in self.vocabulary:
            self.word_to_ix[word] = value
            self.id_to_word[value] = word
            value += 1

    def __len__(self):
        return len(list(self.coco.anns.keys()))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        annotation_id = self.ids[idx]
        description = self.coco.anns[annotation_id]['caption']
        image_id = self.coco.anns[annotation_id]['image_id']
        image_name = self.coco.loadImgs(image_id)[0]['file_name']

        image_name = os.path.join(
                        self.root_dir,
                        image_name)
        image = Image.open(image_name).convert('RGB')
        image = image.resize([224, 224], Image.LANCZOS)
        # image = io.imread(image_name)
        tokens = nltk.tokenize.word_tokenize(str(description).lower())
        caption = []
        caption.append(self.word_to_ix['<start>'])
        caption.extend([self.word_to_ix[token] for token in tokens if token in self.vocabulary])
        caption.append(self.word_to_ix['<end>'])
        

        if self.transform:
            image = self.transform(image)
        captions = torch.Tensor(caption)
        # print(tokens)
        return image, captions, tokens

def generate_batch(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images = [entry[0] for entry in batch]
    captions = [entry[1] for entry in batch]
    descriptions = [entry[1] for entry in batch]

    lengths = [len(cap) for cap in captions]

    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]   

    images = torch.stack(images)

    return images, targets, descriptions

def get_data_loader(dataset, batch_size=128):
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=batch_size,
                                              collate_fn=generate_batch,
                                              num_workers=2,
                                              shuffle=True)
    return data_loader
