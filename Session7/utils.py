import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def plot_loss(train_loss_list, test_loss_list):
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(train_loss_list, label="Train Loss")
    axs.plot(test_loss_list, label="Test Loss")
    axs.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def plot_accuracy(train_accuracy_list, test_accuracy_list):
    fig, axs = plt.subplots(figsize=(5,5))
    axs.plot(train_accuracy_list, label="Train Accuracy")
    axs.plot(test_accuracy_list, label="Test Accuracy")
    axs.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()