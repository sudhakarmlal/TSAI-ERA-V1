import matplotlib.pyplot as plt


def plot_loss_accuracy(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def plot_accuracy(train_acc, test_acc):	
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(train_acc, label="Train Accuracy")
    axs.plot(test_acc, label="Test Accuracy")
    axs.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
	
def plot_max_lr_vs_test_accuracy(max_lr_list,test_accuracy_list):
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(max_lr_list,test_accuracy_list)
    plt.xlabel("Maximum Learning Rate")
    plt.ylabel("Test Accuracy")
    plt.show()
	
