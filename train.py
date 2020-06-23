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

def main(num_epochs=10, embedding_dim=256, data_dir="data/"):
    """ Function to train the model.
    
    Args:
        num_epochs: int
            Number of full dataset iterations to train the model.
        embedding_dim: int
            Output of the CNN model and input of the LSTM embedding size.
        data_dir: str
            Path to the folder of the data.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"WORKING WITH: {device}")

    # Define the paths for train and validation
    train_json_path = data_dir + "annotations/captions_train2014.json"
    train_root_dir = data_dir + "train2014"
    valid_json_path = data_dir + "annotations/captions_val2014.json"
    valid_root_dir = data_dir + "val2014"

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    

    train_dataset = CocoDataset(
                        json_path=train_json_path,
                        root_dir=train_root_dir,
                        transform=transform)

    train_coco_dataset = get_data_loader(
                            train_dataset,
                            batch_size=128)

    valid_dataset = CocoDataset(
                        json_path=valid_json_path,
                        root_dir=valid_root_dir,
                        transform=transform)

    valid_coco_dataset = get_data_loader(
                            valid_dataset,
                            batch_size=1)

    encoder = FeatureExtractor(embedding_dim).to(device)
    decoder = CaptionGenerator(embedding_dim, 512, len(train_dataset.vocabulary), 1).to(device)


    criterion = nn.CrossEntropyLoss()
    # params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=0.01)

    print(f"TRAIN DATASET: {len(train_coco_dataset)}")
    print(f"VALID DATASET: {len(valid_coco_dataset)}")

    total_step = len(train_coco_dataset)
    for epoch in range(num_epochs):
        encoder.train()
        decoder.train()
        train_loss = 0.0
        valid_loss = 0.0
        for i, (images, captions, descriptions) in enumerate(train_coco_dataset):

            # targets = pack_padded_sequence(caption, 0, batch_first=True)[0]
            
            images = images.to(device)
            captions = captions.to(device)
            # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions)

            loss = criterion(outputs.view(-1, len(train_dataset.vocabulary)), captions.view(-1))
            # bleu = calculate_bleu(decoder, features, descriptions, coco_dataset)
            # print(bleu)

            encoder.zero_grad()
            decoder.zero_grad()
            
            loss.backward()
            optimizer.step()

            # Print log info
            train_loss += loss.item()

            '''
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
            '''
                
            # Save the model checkpoints
            if (i+1) % 1000 == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    "models", 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    "models", 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))
        encoder.eval()
        decoder.eval()
        bleu = 0.0
        for i, (images, captions, descriptions) in enumerate(valid_coco_dataset):
            if( i > 80000):
                break
            images = images.to(device)
            captions = captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions)
            loss = criterion(outputs.view(-1, len(train_dataset.vocabulary)), captions.view(-1))
            valid_loss += loss.item()
            bleu += calculate_bleu(decoder, features, descriptions, train_coco_dataset)
        # print(f"BLEU: {bleu / 10000}")
        print("Epoch: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, BLEU: {:.4f}"
             .format(epoch,
                     train_loss / len(train_coco_dataset),
                     valid_loss / 80000,
                     bleu / 80000))


if __name__ == '__main__':
    main(num_epochs=30)