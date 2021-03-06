from pycocotools.coco import COCO
from torchvision.datasets import CocoCaptions
import matplotlib.pyplot as plt


image_dir = "./data/"

json_path =  image_dir + "annotations/captions_train2014.json"
'''
coco = COCO(json_path)
img, target, description = coco[3]
print(target)
imgplot = plt.imshow(img)
plt.show()
'''

import torchvision.transforms as transforms
cap = CocoCaptions(root = image_dir+"train2014/",
                        annFile = json_path)

print('Number of samples: ', len(cap))
img, target = cap[2] # load 4th sample

print(target)
imgplot = plt.imshow(img)
plt.show()

'''

print(target)
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

imgplot = plt.imshow(img)
plt.show()
'''
'''
from dataset import CocoDataset

coco = CocoDataset(json,
image_dir+"train2014/")

img, target, description = coco[3]
# print(target)
print(description)
import matplotlib.pyplot as plt
imgplot = plt.imshow(img)
plt.show()
'''
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
for t in target:
    print(t)

from skimage import io, transform

path = image_dir+"train2014/COCO_train2014_000000095753.jpg" 
image = io.imread(path)
imgplot = plt.imshow(image)
plt.show()
'''