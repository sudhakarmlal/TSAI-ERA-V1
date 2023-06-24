import torch
from torchvision import datasets, transforms


def get_train_loader(batch_size,kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader
 
def get_train_loader_rotation(batch_size,rotation_range,kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomRotation(rotation_range, fill=(1,)),                          
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return train_loader

def get_test_loader(batch_size,kwargs):

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                        ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    return test_loader
def get_classes():
    class_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return class_list	