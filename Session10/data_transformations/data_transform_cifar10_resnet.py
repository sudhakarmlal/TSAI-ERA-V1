from PIL import Image
import cv2
import numpy as np
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,GaussNoise
from albumentations.augmentations.transforms import Cutout,ElasticTransform
from albumentations.pytorch import ToTensor


class album_Compose_train:
    def __init__(self):
        self.transform = Compose(
        [Cutout(num_holes=1, max_h_size=8, max_w_size=8,  fill_value=[0.4914*255, 0.4822*255, 0.4465*255]),
         HorizontalFlip(p=0.1),
         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
         GaussNoise(p=0.15),
         ElasticTransform(p=0.15),
        Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010))),
        ToTensor(),
        ])
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

class album_Compose_test:
    def __init__(self):
        self.transform = Compose(
        [
        Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010))),
        ToTensor(),
        ])
    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img
		
		
def get_train_transform():
    transform = album_Compose_train()
    return transform

def get_test_transform():
    transform = album_Compose_test()
    return transform