# CIFAR-10 Classification with Albumentations and Depthwise Separable Convolutions

This repository provides code implementation for training a CIFAR-10 image classifier using Albumentations for Data augmentation and Depthwise separable convolutions in PyTorch.

## Albumentations:

Albumentations is a popular Python library for image augmentation in machine learning and computer vision tasks. It provides a wide range of transformation techniques, such as random cropping, rotations, flips, color adjustments, and many more. Albumentations is designed to be fast and efficient, making it suitable for real-time applications and large-scale datasets.

Using Albumentations, we can easily apply complex augmentations to our training dataset, improving the model's ability to generalize and handle various real-world scenarios.

## Depthwise Separable Convolutions:

Depthwise separable convolutions are a type of convolutional neural network (CNN) operation that aims to reduce the computational complexity and model size while maintaining or improving performance. It achieves this by decomposing the standard convolution into two separate operations: depthwise convolution and pointwise convolution.

## File Structure
The code repository has the following file structure:

  - `dataset.py` : This code provides a `DataSet` class that serves as a base class for handling image datasets. It includes functionality for loading and transforming data, as well as displaying examples from the dataset.
    
     ![image](https://github.com/Shashank-Gottumukkala/ERA-S9-Albumentations/assets/59787210/5484dab0-0725-4efe-8a63-6d3754cad880)

- `model.py` : This code provides different Convolutional Neural Network (CNN) models implemented using PyTorch,
  - `ConvLayer`: A class representing a convolutional layer with optional skip connections and depthwise separable convolutions. It applies a 3x3 convolution followed by batch normalization, ReLU activation, and       optional dropout.
  -  `Model`: The main model class that combines multiple ConvLayer and transition (downsampling) layers. It ends with a global average pooling layer, a 1x1 convolution, a flattening layer, and a log-softmax            activation.
-  `utis.py` :
     - This code contains classes `Train`, `Test`, `Experiment` that are designed to facilitate training, testing, and conducting experiments with a given model and dataset.

### 3. Model Performance
   - No of Params: `197,418`
   - Best Training Accuracy : `80.6`
   - Best Test Accuracy : `~84`

   - ![image](https://github.com/Shashank-Gottumukkala/ERA-S9-Albumentations/assets/59787210/154e29f5-af47-48b7-b842-ba344d8e5820)
   - ![image](https://github.com/Shashank-Gottumukkala/ERA-S9-Albumentations/assets/59787210/0cd40b5b-0026-4f23-88b7-0d99836ffc06)

  
  


