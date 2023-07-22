#from PIL import Image
import cv2
import numpy as np
from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,GaussNoise, PadIfNeeded,ShiftScaleRotate, CoarseDropout,ToGray
from albumentations.augmentations.dropout import Cutout
from albumentations.pytorch import ToTensorV2


class album_Compose_train:
    def __init__(self):
        self.transform = Compose(
        [
         PadIfNeeded(min_height=48, min_width=48, border_mode=cv2.BORDER_CONSTANT, value=[0.4914*255, 0.4822*255, 0.4465*255], p=1.0),
         RandomCrop(32,32, p=1.0),
         Cutout(num_holes=1, max_h_size=8, max_w_size=8,  fill_value=[0.4914*255, 0.4822*255, 0.4465*255]),
         HorizontalFlip(p=0.2),
         #GaussNoise(p=0.15),
         #ElasticTransform(p=0.15),
        Normalize((0.4914, 0.4822, 0.4465), ((0.2023, 0.1994, 0.2010))),
        ToTensorV2(),
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
        ToTensorV2(),
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