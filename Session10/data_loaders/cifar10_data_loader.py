import torch
from torchvision.datasets import CIFAR10
import torchvision
import torchvision.transforms as transforms

def get_train_loader(batch_size, transform): 
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    return trainloader	
	
def get_test_loader(batch_size, transform): 
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,  download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,  shuffle=False, num_workers=2)										
    return testloader

def get_classes():
    class_list = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return class_list	
