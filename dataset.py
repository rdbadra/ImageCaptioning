import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from vocabulary import create_vocabulary
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
        self.vocabulary, self.ids = create_vocabulary(json_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(list(self.coco.anns.keys()))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        annotation_id = self.ids[idx]
        caption = self.coco.anns[annotation_id]['caption']
        image_id = self.coco.anns[annotation_id]['image_id']
        image_name = self.coco.loadImgs(image_id)[0]['file_name']

        image_name = os.path.join(
                        self.root_dir,
                        image_name)
        image = io.imread(image_name)
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        

        if self.transform:
            sample = self.transform(image)

        sample = {'image': image, 'tokens': tokens}

        return sample