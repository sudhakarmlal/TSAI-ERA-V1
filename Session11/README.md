# ERA_V1 Assignment 11
The objective of the assignment is to train ResNet18 on CIFAR 10 dataset for 20 epochs using OneCycleLR Policy

-  Architecture
   -  The model used for training is ResNet18.
  
   -  Model Summary
      ```
      ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
      ================================================================
                  Conv2d-1           [-1, 64, 18, 18]           1,728
                    ReLU-2           [-1, 64, 18, 18]               0
             BatchNorm2d-3           [-1, 64, 18, 18]             128
               MaxPool2d-4             [-1, 64, 8, 8]               0
                  Conv2d-5             [-1, 64, 8, 8]          36,864
             BatchNorm2d-6             [-1, 64, 8, 8]             128
                    ReLU-7             [-1, 64, 8, 8]               0
                  Conv2d-8             [-1, 64, 8, 8]          36,864
             BatchNorm2d-9             [-1, 64, 8, 8]             128
                   ReLU-10             [-1, 64, 8, 8]               0
                  Block-11             [-1, 64, 8, 8]               0
                 Conv2d-12             [-1, 64, 8, 8]          36,864
            BatchNorm2d-13             [-1, 64, 8, 8]             128
                   ReLU-14             [-1, 64, 8, 8]               0
                 Conv2d-15             [-1, 64, 8, 8]          36,864
            BatchNorm2d-16             [-1, 64, 8, 8]             128
                   ReLU-17             [-1, 64, 8, 8]               0
                  Block-18             [-1, 64, 8, 8]               0
                 Conv2d-19            [-1, 128, 8, 8]          73,728
            BatchNorm2d-20            [-1, 128, 8, 8]             256
                   ReLU-21            [-1, 128, 8, 8]               0
                 Conv2d-22            [-1, 128, 8, 8]         147,456
            BatchNorm2d-23            [-1, 128, 8, 8]             256
                 Conv2d-24            [-1, 128, 8, 8]           8,192
            BatchNorm2d-25            [-1, 128, 8, 8]             256
                   ReLU-26            [-1, 128, 8, 8]               0
                  Block-27            [-1, 128, 8, 8]               0
                 Conv2d-28            [-1, 128, 8, 8]         147,456
            BatchNorm2d-29            [-1, 128, 8, 8]             256
                   ReLU-30            [-1, 128, 8, 8]               0
                 Conv2d-31            [-1, 128, 8, 8]         147,456
            BatchNorm2d-32            [-1, 128, 8, 8]             256
                   ReLU-33            [-1, 128, 8, 8]               0
                  Block-34            [-1, 128, 8, 8]               0
                 Conv2d-35            [-1, 256, 8, 8]         294,912
            BatchNorm2d-36            [-1, 256, 8, 8]             512
                   ReLU-37            [-1, 256, 8, 8]               0
                 Conv2d-38            [-1, 256, 8, 8]         589,824
            BatchNorm2d-39            [-1, 256, 8, 8]             512
                 Conv2d-40            [-1, 256, 8, 8]          32,768
            BatchNorm2d-41            [-1, 256, 8, 8]             512
                   ReLU-42            [-1, 256, 8, 8]               0
                  Block-43            [-1, 256, 8, 8]               0
                 Conv2d-44            [-1, 256, 8, 8]         589,824
            BatchNorm2d-45            [-1, 256, 8, 8]             512
                   ReLU-46            [-1, 256, 8, 8]               0
                 Conv2d-47            [-1, 256, 8, 8]         589,824
            BatchNorm2d-48            [-1, 256, 8, 8]             512
                   ReLU-49            [-1, 256, 8, 8]               0
                  Block-50            [-1, 256, 8, 8]               0
                 Conv2d-51            [-1, 512, 8, 8]       1,179,648
            BatchNorm2d-52            [-1, 512, 8, 8]           1,024
                   ReLU-53            [-1, 512, 8, 8]               0
                 Conv2d-54            [-1, 512, 8, 8]       2,359,296
            BatchNorm2d-55            [-1, 512, 8, 8]           1,024
                 Conv2d-56            [-1, 512, 8, 8]         131,072
            BatchNorm2d-57            [-1, 512, 8, 8]           1,024
                   ReLU-58            [-1, 512, 8, 8]               0
                  Block-59            [-1, 512, 8, 8]               0
                 Conv2d-60            [-1, 512, 8, 8]       2,359,296
            BatchNorm2d-61            [-1, 512, 8, 8]           1,024
                   ReLU-62            [-1, 512, 8, 8]               0
                 Conv2d-63            [-1, 512, 8, 8]       2,359,296
            BatchNorm2d-64            [-1, 512, 8, 8]           1,024
                   ReLU-65            [-1, 512, 8, 8]               0
                  Block-66            [-1, 512, 8, 8]               0
      AdaptiveAvgPool2d-67            [-1, 512, 1, 1]               0
                 Conv2d-68             [-1, 10, 1, 1]           5,130
      ================================================================
      Total params: 11,173,962
      Trainable params: 11,173,962
      Non-trainable params: 0
      ----------------------------------------------------------------
      Input size (MB): 0.01
      Forward/backward pass size (MB): 7.95
      Params size (MB): 42.63
      Estimated Total Size (MB): 50.58
      ----------------------------------------------------------------
      ```
       

-  Data Augmentation
   -  Albumentation Library is used to apply transforms to the data
   -  Transforms applied are: RandomCrop of (32, 32) after padding of 4 >> FlipLR >> Followed by CutOut(8, 8)
   -  Images after augmentation
  ![data](./images/augdata.png)

 
-  Learning Rate
   -  The maximum learning was found using the LRFinder library.
  ![LRFinder](./images/LRFinder.png)

   -  The LR was then set to achieve its maximum at 5th epoch
  ![Lr](./images/LR.png)

-  Loss and Accuracy
   -  The model acheives the target of 90% test accuracy at 24th epoch
  ![log](./images/Log.png)

   -  Comparison of train and test loss and accuracy
  ![AcuracyLoss](./images/AccuracyLoss.png)

   -  Misclassified Images
  ![Missclassified](./images/MisclassifiedImages.png)

  -  GradCam Images
  ![Missclassified](./images/gradcam.png)


