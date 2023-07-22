import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dropout_prob=0.1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), # Output= 32, RF=3, Jump =1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), # Output= 32, RF=5, Jump =1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), # Output= 32, RF=7, Jump =1
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.transition1= nn.Sequential(
                                                                  
                    nn.Conv2d(64, 32, 1, padding=1, bias=False, stride=2), # Output= 16, RF=8, Jump =2
                    nn.ReLU(),
                    nn.BatchNorm2d(32))
                    #nn.Dropout2d(p=dropout_prob))
        self.conv2 = nn.Sequential(           
            nn.Conv2d(32, 32, 3, padding=1, bias=False,dilation=2), # Output= 16, RF=12, Jump =2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(32, 32, 3, padding=1, bias=False), # Output= 16, RF=16, Jump =2
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), # Output= 16, RF=20, Jump =2
            nn.ReLU(),
            nn.BatchNorm2d(64)
            #nn.Dropout2d(p=dropout_prob),

        ) # Input=28, Output=28, rf=3
        self.transition2= nn.Sequential(
                    
                    nn.Conv2d(64, 32, 1, padding=1, bias=False, stride=2), # Output= 8, RF=22, Jump =4
                    nn.ReLU(),
                    nn.BatchNorm2d(32))
                    #nn.Dropout2d(p=dropout_prob))
        
        self.conv3 = nn.Sequential(           
            nn.Conv2d(32, 32, 3, padding=1, bias=False, groups=32),  # Output= 8, RF=30, Jump =4
            nn.Conv2d(32, 32, 1, padding=1, bias=False), # Output= 8, RF=30, Jump =4
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), # Output= 8, RF=38, Jump =4
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(64, 96, 3, padding=1, bias=False), # Output= 8, RF=46, Jump =4
            nn.ReLU(),
            nn.BatchNorm2d(96))
            #nn.Dropout2d(p=dropout_prob))
        
        self.transition3= nn.Sequential(
                    
                    nn.Conv2d(96, 32, 1, padding=1, bias=False, stride=2),  # Output= 4, RF=50, Jump =8
                    nn.ReLU(),
                    nn.BatchNorm2d(32))
                    #nn.Dropout2d(p=dropout_prob))
        
        self.conv4 = nn.Sequential(           
            nn.Conv2d(32, 64, 3, padding=1, bias=False), # Output= 4, RF=66, Jump =8
            nn.ReLU(),
            nn.BatchNorm2d(64),
            #nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(64, 32, 3, padding=1, bias=False), # Output= 4, RF=82, Jump =8
            nn.ReLU(),
            #nn.BatchNorm2d(32),
            #nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(32, 10, 3, padding=1, bias=False) # Output= 4, RF=98, Jump =8
           )
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Output= 4, RF=122, Jump =8
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.transition1(x)
        x = self.conv2(x)
        x = self.transition2(x)
        x = self.conv3(x)
        x = self.transition3(x)
        x = self.conv4(x)

                   
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
