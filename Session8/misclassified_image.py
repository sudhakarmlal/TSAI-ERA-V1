import matplotlib.pyplot as plt
import numpy as np

def imshow1(img):
    img = img / 2 + 0.5     # unnormalize
    #npimg = img.numpy()
    plt.imshow(np.transpose(img, (1,2, 0)))


def display_misclassfied_cifar10_images(testloader, model, device, classes, num_display=25):
    incorrect_image_list =[]
    predicted_label_list =[]
    correct_label_list = []
    for (i, [data, target]) in enumerate(testloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True).squeeze(1)         
        idxs_mask = (pred !=  target).view(-1)
        img_nm = data[idxs_mask].cpu().numpy()
        img_nm = img_nm.reshape(img_nm.shape[0], 3, 32, 32)
        if img_nm.shape[0] > 0:
            img_list = [img_nm[i] for i in range(img_nm.shape[0])]
            incorrect_image_list.extend(img_list)
            predicted_label_list.extend(pred[idxs_mask].detach().cpu().numpy())
            correct_label_list.extend(target[idxs_mask].detach().cpu().numpy())


    plt.figure(figsize=(15,15))
    columns = 5
    i= 0
    # Display the list of 25 misclassified images
    for index, image in enumerate(incorrect_image_list) :
        ax = plt.subplot(5, 5, i+1)
        ax.set_title("Actual: " + str(classes[correct_label_list[index]]) + ", Predicted: " + str(classes[predicted_label_list[index]]))
        ax.axis('off')
    #plt.imshow(image)
        imshow1(image)
        i +=1
        if i==num_display:
            break