import os
from PIL import Image  
from collections import Counter
import nltk
nltk.download('punkt')

def create_vocabulary(coco):
    ''' Gets all the words from the COCO API to generate the vocabulary 
    
    Args:
        coco: COCO
            Instance of COCO, contains the relation between caption and image.
    Returns:
        vocabulary: list
            List containing all the words in lowercase in the vocabulary that appear
            ar least 5 times.
        ids: list
            List of containing the indices of all the captions
    '''
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
