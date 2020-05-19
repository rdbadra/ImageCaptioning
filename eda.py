import os
from PIL import Image  

image_dir = "/home/roberto/Documentos/TFM-UOC/pytorch-tutorial/tutorials/03-advanced/image_captioning/data/train2014"

images = os.listdir(image_dir)
print(images[0])
image = images[0]
path = os.path.join(image_dir, image)

im = Image.open(path)  
  
im.show() 