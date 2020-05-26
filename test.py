from pycocotools.coco import COCO
from torchvision.datasets import CocoCaptions


image_dir = "/home/roberto/Documentos/TFM-UOC/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/"

json =  image_dir + "annotations/captions_train2014.json"

# self.coco = COCO(json_path)
'''

import torchvision.transforms as transforms
cap = CocoCaptions(root = image_dir+"train2014/",
                        annFile = json)

print('Number of samples: ', len(cap))
img, target = cap[20] # load 4th sample


print(target)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

imgplot = plt.imshow(img)
plt.show()
'''
from dataset import CocoDataset

coco = CocoDataset(json,
image_dir+"train2014/")

img, target = coco[0]
print(target)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
for t in target:
    print(t)

from skimage import io, transform

path = image_dir+"train2014/COCO_train2014_000000095753.jpg" 
image = io.imread(path)
imgplot = plt.imshow(image)
plt.show()