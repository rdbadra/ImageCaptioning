import os
from PIL import Image  
from collections import Counter
import nltk
nltk.download('punkt')


image_dir = "/home/roberto/Documentos/TFM-UOC/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/"

json =  image_dir + "annotations/captions_train2014.json"

'''
print(len(ids))
print(list(ids)[0])
caption = str(coco.anns[48]['caption'])
print(caption)
tokens = nltk.tokenize.word_tokenize(caption.lower())
print(tokens)
counter.update(tokens)
print(counter)
'''

def create_vocabulary(coco):
    print("creating vocab")
    counter = Counter()
    ids = list(coco.anns.keys())
    for id in ids:
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
    vocabulary = [word for word, count in counter.items() if count >= 5]
    vocabulary.append('<start>')
    vocabulary.append('<end>')
    return vocabulary, ids


if __name__ == '__main__':
    image_dir = "/home/roberto/Documentos/TFM-UOC/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/"
    json_path =  image_dir + "annotations/captions_train2014.json"
