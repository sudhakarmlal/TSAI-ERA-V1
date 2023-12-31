import numpy as np
import torch
from torchvision import datasets, transforms
import albumentations as A

from .dataset import Dataset

class AlbData(datasets.CIFAR10):
    def __init__(self, root, alb_transforms=None, **kwargs):
        super(AlbData, self).__init__(root, **kwargs)
        self.alb_transforms = alb_transforms

    def __getitem__(self, index):
        image, label = super(AlbData, self).__getitem__(index)
        if self.alb_transforms is not None:
            image = self.alb_transforms(image=np.array(image))['image']
        return image, label
    
class CIFAR10(Dataset):
        mean = (0.49139968, 0.48215827, 0.44653124)
        std = (0.24703233, 0.24348505, 0.26158768)
        classes = None
    
        def get_train_transforms(self):
            if self.alb_transforms is None:
                self.alb_transforms = [
                A.HorizontalFlip(p = 0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
                A.CoarseDropout(max_holes= 1, max_height=16, max_width=16, p=0.2, fill_value=0),
                
                # A.Downscale(0.8, 0.95, p =0.2),
                # A.ToGray(p = 0.5),
                
            ]
            return super(CIFAR10, self).get_train_transforms()
        
        def get_train_loader(self):
            super(CIFAR10, self).get_train_loader()

            train_data = AlbData('../data', train = True, download = True, alb_transforms= self.train_transforms)
            if self.classes is None:
                self.classes = {i: c for i, c in enumerate(train_data.classes)}
            self.train_loader = torch.utils.data.DataLoader(train_data, shuffle=self.shuffle, **self.loader_kwargs)
            return self.train_loader 

        def get_test_loader(self):
            super(CIFAR10, self).get_test_loader()

            test_data = AlbData('../data', train=False, download=True, alb_transforms=self.test_transforms)
            self.test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **self.loader_kwargs)
            return self.test_loader     

        def show_transform(self, img):
            return img.permute(1, 2, 0)
            
