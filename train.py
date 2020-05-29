from dataset import get_data_loader, CocoDataset
import matplotlib.pyplot as plt
from model import FeatureExtractor, CaptionGenerator
import torch.optim as optim
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import os
from torchvision import transforms
from tqdm import tqdm
from evaluation import calculate_bleu

def main(num_epochs=1, data_dir="data/"):
    print("main")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"WORKING WITH: {device}")
    # image_dir = "/home/roberto/Documentos/TFM-UOC/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/"
    # json_path =  image_dir + "annotations/captions_train2014.json"
    json_path = "data/annotations/captions_train2014.json"


    root_dir = "data/train2014"
    '''
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),}
    '''
    '''
    image_datasets = {x: datasets.ImageFolder(os.path.join(image_dir, 'train2014'),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: CocoDataset(json_path=image_datasets[x])
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    '''
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    

    dataset = CocoDataset(
                        json_path=json_path,
                        root_dir=root_dir,
                        transform=transform)

    coco_dataset = get_data_loader(
                        dataset,
                        batch_size=128)

    encoder = FeatureExtractor(256).to(device)
    decoder = CaptionGenerator(256, 512, len(dataset.vocabulary), 1).to(device)


    criterion = nn.CrossEntropyLoss()
    # params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=0.01)

    total_step = len(coco_dataset)
    for epoch in range(num_epochs):
        for i, (images, captions, descriptions) in enumerate(coco_dataset):

            # targets = pack_padded_sequence(caption, 0, batch_first=True)[0]
            
            images = images.to(device)
            captions = captions.to(device)
            # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions)

            loss = criterion(outputs.view(-1, len(dataset.vocabulary)), captions.view(-1))
            # bleu = calculate_bleu(decoder, features, descriptions, coco_dataset)
            # print(bleu)

            encoder.zero_grad()
            decoder.zero_grad()
            
            loss.backward()
            optimizer.step()

            # Print log info

            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % 1000 == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    "models", 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    "models", 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            


    

if __name__ == '__main__':
    main(num_epochs=5)