import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
import os

def train_test_loader(BATCH_SIZE,get_train_transform, get_test_transform,get_train_loader, get_test_loader, get_classes):
  transform_train = get_train_transform()
  transform_test = get_test_transform()

  trainloader = get_train_loader(BATCH_SIZE, transform_train)
  testloader = get_test_loader(BATCH_SIZE, transform_test)
  classes = get_classes()
  return trainloader, testloader, classes,transform_train,transform_test
  
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def get_model(CustomResNet):
  model =  CustomResNet()
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  model.device = torch.device("cuda" if use_cuda else "cpu")
  model =  CustomResNet().to(model.device)
  return model
 
def find_lr_value(model,EPOCHS_TO_TRY,max_lr_list,test_accuracy_list,PATH_BASE_MODEL,max_lr_finder_schedule,BATCH_SIZE,train,test,trainloader,testloader):
  for lr_value in max_lr_list:
      model.load_state_dict(torch.load(PATH_BASE_MODEL))
      optimizer = optim.SGD(model.parameters(), lr=lr_value/10, momentum=0.9)

      lr_finder_schedule = lambda t: np.interp([t], [0, EPOCHS_TO_TRY], [lr_value/10,  lr_value])[0]
      lr_finder_lambda = lambda it: lr_finder_schedule(it * BATCH_SIZE/50176)
      max_lr_finder = max_lr_finder_schedule(no_of_images=50176, batch_size=BATCH_SIZE, base_lr=lr_value/10, max_lr=lr_value, total_epochs=5)
      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[max_lr_finder])
      train_losses = []
      test_losses = []
      train_acc = []
      test_acc = []
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")      
      for epoch in range(EPOCHS_TO_TRY):
          print("MAX LR:" ,lr_value, " EPOCH:", (epoch+1))        
          train(model, device, trainloader, optimizer, epoch, train_losses,scheduler,train_acc,True )
          test(model, device, testloader, test_losses, test_acc)
      t_acc = test_acc[-1]
      test_accuracy_list.append(t_acc)
      print(" For Max LR: ", lr_value, " Test Accuracy: ", t_acc)
      
def train_model(best_test_accuracy,EPOCHS, model,trainloader,testloader,optimizer,train,test,train_losses,test_losses,scheduler,train_acc,test_acc,PATH):
  for epoch in range(EPOCHS):
      print("EPOCH:", (epoch+1))
      use_cuda = torch.cuda.is_available()
      device = torch.device("cuda" if use_cuda else "cpu")  
      train(model, device, trainloader, optimizer, epoch, train_losses,scheduler,train_acc, True )
      test(model, device, testloader, test_losses, test_acc)
      t_acc = test_acc[-1]
      if t_acc > best_test_accuracy:
          print("Test Accuracy: " + str(t_acc) + " has increased. Saving the model")
          best_test_accuracy = t_acc
          torch.save(model.state_dict(), PATH)