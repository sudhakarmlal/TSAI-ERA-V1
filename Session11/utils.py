from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from torch_lr_finder import LRFinder
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image



def plot_original_data(batch_data, batch_label, num_plots, row, col): 

    fig = plt.figure()

    for i in range(num_plots):
        plt.subplot(row,col,i+1)
        plt.tight_layout()
        img= batch_data[i]
        plt.imshow(img.T)
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])


def plot_loss_accuracy(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
    # axs[0].plot(train_losses,'g', label='Train Losses')
    # axs[0].plot(test_losses, 'b', label='Test Losses')
    # axs[0].set(xlabel='Epochs', ylabel='Losses', title='Train and Test Loss')
    # axs[0].legend()
    # axs[1].plot(train_accuracy,'g', label='Train Accuracy')
    # axs[1].plot(test_accuracy, 'b', label='Test Accuracy')
    # axs[1].set(xlabel='Epochs', ylabel='Accuracy', title='Train and Test Accuracy')
    # axs[1].legend()

def overplot_loss_accuracy(train_losses, test_losses, train_acc, test_acc):
    fig, axs = plt.subplots(1,2,figsize=(15,10))
    axs[0].plot(train_losses,'g', label='Train Losses')
    axs[0].plot(test_losses, 'b', label='Test Losses')
    axs[0].set(xlabel='Epochs', ylabel='Losses', title='Train and Test Loss')
    axs[0].legend()
    axs[1].plot(train_acc,'g', label='Train Accuracy')
    axs[1].plot(test_acc, 'b', label='Test Accuracy')
    axs[1].set(xlabel='Epochs', ylabel='Accuracy', title='Train and Test Accuracy')
    axs[1].legend()
	
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
	
def get_incorrrect_predictions(model, loader, device, criterion):
    """Get all incorrect predictions

    Args:
        model (Net): Trained model
        loader (DataLoader): instance of data loader
        device (str): Which device to use cuda/cpu
        criterion(): Loss function used
    Returns:
        list: list of all incorrect predictions and their corresponding details
    """
    model.eval()
    incorrect = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target).item()
            pred = output.argmax(dim=1)
            for d, t, p, o in zip(data, target, pred, output):
                if p.eq(t.view_as(p)).item() == False:
                    incorrect.append(
                        [d.cpu(), t.cpu(), p.cpu(), o[p.item()].cpu()])

    return incorrect

def plot_incorrect_predictions(predictions, class_map, count=10):
    """Plot Incorrect predictions

    Args:
        predictions (list): List of all incorrect predictions
        class_map (dict): Lable mapping
        count (int, optional): Number of samples to print, multiple of 5. Defaults to 10.
    """
    print(f'Total Incorrect Predictions {len(predictions)}')

    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return

    classes = list(class_map.values())

    fig = plt.figure(figsize=(10, 5))
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{classes[t.item()]}/{classes[p.item()]}')

        #std deviation correction to be made
        std_correction = np.asarray([0.2439, 0.2402, 0.2582]).reshape(3, 1, 1)  #(0.4942, 0.4851, 0.4504), (0.2439, 0.2402, 0.2582)
        #mean correction to be made
        mean_correction = np.asarray([0.4942, 0.4851, 0.4504]).reshape(3, 1, 1)
        #convert the tensor img to numpy img and de normalize
        npimg = np.multiply(d.cpu().numpy(), std_correction) + mean_correction

        plt.imshow((npimg*255).transpose(1, 2, 0).astype(np.uint8))
        # plt.imshow(d.cpu().numpy().transpose(1, 2, 0))
        if i+1 == 5*(count/5):
            break
        
def get_mean_std(data_loader, device):
    """
    This function is used to calculate the mean and standard deviation of the train/test loader

    Args:
        data_loader (DataLoader): Train/Test Loader 
        device (str): device on which to run the computation

    Returns:
            mean: mean of the data set
            stdev: standard deviation of the data set
    """
    mean = 0.0
    stdev = 0.0
    for batch_idx, (data, target) in enumerate(data_loader):
      data = data.to(device)
      mean += torch.mean(data, dim=(0,2,3))
      stdev += torch.std(data, dim=(0,2,3))
    mean = mean/len(data_loader)
    stdev = stdev/len(data_loader)
    return mean, stdev

def denormalize(img):
  channel_means = (0.4914, 0.4822, 0.4465)
  channel_stdevs = (0.2470, 0.2435, 0.2616)
  img = img.astype(dtype=np.float32)

  for i in range(img.shape[0]):
      img[i] = ((img[i] * channel_stdevs[i]) + channel_means[i])/255

  return np.transpose(img, (1, 2, 0))


def visualize_augmentations(batch_data, batch_label, title, class_map, samples=10, cols=5):
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    classes = list(class_map.values())
    figure.suptitle(title)
    for i in range(samples):
        image, t = batch_data[i], batch_label[i]
        npimg = image.numpy()
        ax.ravel()[i].imshow(np.transpose(npimg, (1, 2, 0)))
        ax.ravel()[i].set_title(f'{classes[t]}')
        # ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def print_epoch_progress(epoch, lr, train_accuracy, train_loss, test_accuracy, test_loss):
        """ Log status for epoch

        Args:
            epoch (int): epoch number
            train_correct (int): number or correct predictions
            train_loss (float): loss incurred while training
            valid_correct (int): number of correct predictions
            valid_loss (float): loss incurred while validation
        """
        
        print(f'| {epoch:5} | {str(round(lr, 6)):8} | {str(round(train_accuracy,7)):8}% | {str(round(train_loss, 6)):9} | {str(round(test_accuracy,7)):8}% | {str(round(test_loss, 6)):8} |')

def get_device():
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    return cuda, device

def print_model_summary(model, device, input_size):
    print(summary(model.to(device), input_size=input_size))

def lr_finder(model, train_loader, optimizer, criterion, device):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to

def plot_lr(num_epochs, lr):
    epochs = range(1,num_epochs+1)
    plt.plot(epochs,lr)
    plt.title("Learning Rate")
    plt.show()

def misclassifieddata(incorrect_prdeictions):
    misclsdata, misclstarget, misclspred = [], [], []
    for i in range(10):
        tmpdata, tmptarget, tmppred, _  = incorrect_prdeictions[i]
        misclsdata.append(tmpdata)
        misclstarget.append(tmptarget)
        misclspred.append(tmppred)

    return misclsdata, misclstarget, misclspred

def plot_grad_cam(miscls_data, miscls_label, miscls_pred, model, device, target_layers, title, class_map, samples=10, cols=5):
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device)

    
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    classes = list(class_map.values())
    figure.suptitle(title)
    for i in range(len(miscls_data)):
        input_tensor = miscls_data[i].unsqueeze(0).to(device)
        targets = [ClassifierOutputTarget(miscls_label[i])]
        rgb_img = denormalize(miscls_data[i].cpu().numpy().squeeze())
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        t =  miscls_label[i]
        npimg_ = miscls_data[i].numpy()
        npimg = np.transpose(npimg_,(1,2,0))
        ax.ravel()[i].imshow(visualization/255. + npimg)
        ax.ravel()[i].set_title(f'{classes[t]}')
        # ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
