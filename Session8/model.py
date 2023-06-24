
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchsummary import summary

class model(nn.Module):
    def __init__(self,dropout_prob):
        super(model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=dropout_prob),
            
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(16, 32, 1, padding=1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(32))

        self.pool1= nn.MaxPool2d(2, 2) 

        self.block2= nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=dropout_prob), 
           
           nn.Conv2d(16, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.BatchNorm2d(32),
           nn.Dropout2d(p=dropout_prob),

           nn.Conv2d(32, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.BatchNorm2d(32),
           nn.Dropout2d(p=dropout_prob),  

           nn.Conv2d(32, 32, 1, padding=1, bias=False), 
           nn.ReLU(),
           nn.BatchNorm2d(32)   
        
        ) 


        self.pool2= nn.MaxPool2d(2, 2) 

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=dropout_prob), 
           
           nn.Conv2d(16, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.BatchNorm2d(32),
           nn.Dropout2d(p=dropout_prob), 

           nn.Conv2d(32, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.BatchNorm2d(32),
           nn.Dropout2d(p=dropout_prob),        
        
        ) 

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  
      
        self.transition= nn.Sequential(
            nn.Conv2d(32, 10, 1, bias=False), 4
            nn.ReLU(),
            nn.BatchNorm2d(10))
                    
    
    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.global_avgpool(x)
        x = self.transition(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class model1(nn.Module):
    def __init__(self,dropout_prob):
        super(model1, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout2d(p=dropout_prob),
            
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(16, 32, 1, padding=1, bias=False), 
            nn.ReLU(),
            nn.GroupNorm(4,32))

        self.pool1= nn.MaxPool2d(2, 2) 

        self.block2= nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout2d(p=dropout_prob), 
           
           nn.Conv2d(16, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.GroupNorm(4,32),
           nn.Dropout2d(p=dropout_prob), 

           nn.Conv2d(32, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.GroupNorm(4,32),
           nn.Dropout2d(p=dropout_prob),   

           nn.Conv2d(32, 32, 1, padding=1, bias=False), 
           nn.ReLU(),
           nn.GroupNorm(4,32)   
        
        ) 


        self.pool2= nn.MaxPool2d(2, 2) 

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout2d(p=dropout_prob), 
           
           nn.Conv2d(16, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.GroupNorm(4,32),
           nn.Dropout2d(p=dropout_prob), 

           nn.Conv2d(32, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.GroupNorm(4,32),
           nn.Dropout2d(p=dropout_prob),        
        
        ) 

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  
      
        self.transition= nn.Sequential(
            nn.Conv2d(32, 10, 1, bias=False), 
            
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.global_avgpool(x)
        x = self.transition(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)

class model2(nn.Module):
    def __init__(self,dropout_prob):
        super(model2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3,padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm((16,32,32),elementwise_affine=False),
            nn.Dropout2d(p=dropout_prob),
            
            nn.Conv2d(16, 16, 3, padding=1,bias=False),
            nn.ReLU(),
            nn.LayerNorm((16, 32, 32),elementwise_affine=False),
            nn.Dropout2d(p=dropout_prob),

            nn.Conv2d(16, 32, 1, bias=False), 
            nn.ReLU(),
            nn.LayerNorm((32, 32, 32),elementwise_affine=False)
        )

        self.pool1= nn.MaxPool2d(2, 2) 

        self.block2= nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm((16, 16, 16),elementwise_affine=False),
            nn.Dropout2d(p=dropout_prob),
           
           nn.Conv2d(16, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.LayerNorm((32, 16, 16),elementwise_affine=False),
           nn.Dropout2d(p=dropout_prob), 

           nn.Conv2d(32, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.LayerNorm((32, 16, 16),elementwise_affine=False),
           nn.Dropout2d(p=dropout_prob),  

           nn.Conv2d(32, 32, 1, bias=False), 
           nn.ReLU(),
           nn.LayerNorm((32, 16, 16),elementwise_affine=False)   
        
        ) 


        self.pool2= nn.MaxPool2d(2, 2) 

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.LayerNorm((16, 8, 8),elementwise_affine=False),
            nn.Dropout2d(p=dropout_prob), 
           
           nn.Conv2d(16, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.LayerNorm((32, 8, 8),elementwise_affine=False),
           nn.Dropout2d(p=dropout_prob), 

           nn.Conv2d(32, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.LayerNorm((32, 8, 8),elementwise_affine=False),
           nn.Dropout2d(p=dropout_prob),        
        
        ) 

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  
      
        self.transition= nn.Sequential(
            nn.Conv2d(32, 10, 1, bias=False), 
            
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.global_avgpool(x)
        x = self.transition(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
class model3(nn.Module):
    def __init__(self,dropout_prob):
        super(model3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=dropout_prob)
        )
            
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(p=dropout_prob)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )

        self.pool1= nn.MaxPool2d(2, 2) 

        self.conv4= nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=7, Output=5, rf=24
           
        self.conv5= nn.Sequential(
           nn.Conv2d(16, 16, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.BatchNorm2d(16),
           nn.Dropout2d(p=dropout_prob)
        ) # Input=7, Output=5, rf=24

        self.conv6= nn.Sequential(
           nn.Conv2d(16, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.BatchNorm2d(32),
           nn.Dropout2d(p=dropout_prob)
         ) # Input=7, Output=5, rf=24  

        self.conv7= nn.Sequential(
           nn.Conv2d(32, 32, 1, padding=1, bias=False), 
           nn.ReLU(),
           nn.BatchNorm2d(32)   
        
        ) 


        self.pool2= nn.MaxPool2d(2, 2) 

        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=dropout_prob)
        ) 
           
        self.conv9= nn.Sequential(
           nn.Conv2d(16, 16, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.BatchNorm2d(16),
           nn.Dropout2d(p=dropout_prob)
        ) 

        self.conv10= nn.Sequential(
           nn.Conv2d(16, 32, 3, padding=1, bias=False),
           nn.ReLU(),
           nn.BatchNorm2d(32),
           nn.Dropout2d(p=dropout_prob)        
        
        ) 

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  
      
        self.transition= nn.Sequential(
            nn.Conv2d(32, 10, 1, bias=False), 
            nn.ReLU(),
            nn.BatchNorm2d(10))
                    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.conv3(x)
        x = self.pool1(x)
        #print(x.shape)
        x = self.conv4(x)
        #print(x.shape)
        x = x + self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool2(x)
        x = self.conv8(x)
        x = x + self.conv9(x)
        x = self.conv10(x)
        x = self.global_avgpool(x)
        x = self.transition(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
    
class model4(nn.Module):
    def __init__(self,dropout_prob):
        super(model4, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=14, Output=14, rf=10
        self.conv4 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=14, Output=14, rf=14

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=7, Output=5, rf=24
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(14, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
class model5(nn.Module):
    def __init__(self,dropout_prob):
        super(model5, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=14, Output=14, rf=10
        self.conv4 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=14, Output=14, rf=14

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=7, Output=5, rf=24
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(14, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
class model6(nn.Module):
    def __init__(self,dropout_prob):
        super(model6, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=10
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=14
        )

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob) # Input=7, Output=5, rf=24
        )
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(14, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
class model7(nn.Module):
    def __init__(self,dropout_prob):
        super(model7, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=10
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=14
        )

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob) # Input=7, Output=5, rf=24
        )
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(14, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
class model8(nn.Module):
    def __init__(self,dropout_prob):
        super(model8, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=10
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=14
        )

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob) # Input=7, Output=5, rf=24
        )
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(14, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
class model9(nn.Module):
    def __init__(self,dropout_prob):
        super(model9, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=10
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=14
        )

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob) # Input=7, Output=5, rf=24
        )
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(12, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
class model10(nn.Module):
    def __init__(self,dropout_prob):
        super(model10, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=10
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=14
        )

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob) # Input=7, Output=5, rf=24
        )
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(12, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    

class model11(nn.Module):
    def __init__(self,dropout_prob):
        super(model11, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=10
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob) # Input=14, Output=14, rf=14
        )

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(p=dropout_prob) # Input=7, Output=5, rf=24
        )
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(12, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)

class model12(nn.Module):
    def __init__(self,dropout_prob):
        super(model1, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU()
        ) # Input=28, Output=28, rf=3

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        ) # Input=14, Output=14, rf=10
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU()
        ) # Input=14, Output=14, rf=14

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=0),
            nn.ReLU(),
        ) # Input=7, Output=5, rf=24
    
        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 10, 3, padding=0),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
    
class model13(nn.Module):
    def __init__(self,dropout_prob):
        super(model2, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 14, 3, padding=1, bias=False),
            nn.ReLU()
        ) # Input=28, Output=28, rf=3

        self.conv2 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU()
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU()
        ) # Input=14, Output=14, rf=10
        self.conv4 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU()
        ) # Input=14, Output=14, rf=14

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=0, bias=False),
            nn.ReLU(),
        ) # Input=7, Output=5, rf=24
    
        self.conv6 = nn.Sequential(
            nn.Conv2d(14, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)

class model14(nn.Module):
    def __init__(self,dropout_prob):
        super(model3, self).__init__()
        #dropout_prob=dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=3
        self.conv2 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=28, Output=28, rf=5

        self.pool1= nn.MaxPool2d(2, 2) # Input=28, Output=14, rf=6

        self.conv3 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=14, Output=14, rf=10
        self.conv4 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob)
        ) # Input=14, Output=14, rf=14

        self.pool2= nn.MaxPool2d(2, 2) # Input=14, Output=7, rf=16

        self.conv5 = nn.Sequential(
            nn.Conv2d(14, 14, 3, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout2d(p=dropout_prob) 
        ) # Input=7, Output=5, rf=24
       
        self.conv6 = nn.Sequential(
            nn.Conv2d(14, 10, 3, padding=0, bias=False),
        ) # Input=5, Output=3, rf=32

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # Input=3, Output=1, rf=40
      
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)               
        x = self.global_avgpool(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)

    
def get_train(model, device, train_loader, optimizer, epoch,train_losses,train_acc):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  loss_list = []
  acc_list = []
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    loss_list.append(loss.detach().numpy())
    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    acc_list.append(100*correct/processed)
  train_losses.append(sum(loss_list)/len(loss_list))
  train_acc.append(sum(acc_list)/len(acc_list))

  return train_losses,train_acc

def get_test(model, device, test_loader,test_losses,test_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))
    
    return test_losses,test_acc

def print_model_summary(dropout_prob, inputsize,Net):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(dropout_prob=dropout_prob).to(device)
    summary(model, input_size=inputsize)