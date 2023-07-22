import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNBlock(nn.Module):
    

    def __init__(self, in_planes, planes, stride=1, p=0.0):
        super(ConvBNBlock, self).__init__()
        self.dropout_prob=p
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.drop_out = nn.Dropout2d(p=self.dropout_prob)        

    def forward(self, x):
        out = F.relu(self.drop_out(self.bn(self.conv(x))))
        return out

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, p=0.0):
        super(TransitionBlock, self).__init__()
        self.p = p
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(planes)
        self.drop_out = nn.Dropout2d(p=self.p)    
       

    def forward(self, x):
        x = F.relu(self.drop_out(self.bn(self.max_pool(self.conv(x)))))
         
        return x


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, p=0.0):
        super(ResBlock, self).__init__()
        self.p = p
        self.transition_block = TransitionBlock(in_planes, planes, stride, p)
        self.conv_block1 = ConvBNBlock(planes, planes, stride, p)
        self.conv_block2 = ConvBNBlock(planes, planes, stride, p)
  

    def forward(self, x):
        x = self.transition_block(x)
        r = self.conv_block2(self.conv_block1(x))
        out = x + r
        
         
        return out


class CustomResNet(nn.Module):
    def __init__(self, p=0.0, num_classes=10):
        super(CustomResNet, self).__init__()
        self.in_planes = 64
        self.p = p

        self.conv = ConvBNBlock(3, 64, 1, p)
        self.layer1 = ResBlock(64,128, 1, p)
        self.layer2 = TransitionBlock(128, 256, 1, p)
        self.layer3 =  ResBlock(256,512, 1, p)
        self.max_pool = nn.MaxPool2d(4, 4)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.max_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out)
