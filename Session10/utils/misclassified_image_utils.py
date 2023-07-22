import matplotlib.pyplot as plt
import numpy as np

def imshow1(img):
    img = img / 2 + 0.5     # unnormalize
    #npimg = img.numpy()
    plt.imshow(np.transpose(img, (1,2, 0)))
    
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(img, (1,2, 0)))    


def display_misclassfied_ciphar10_images(testloader, model, device, classes, num_display=25):
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
    return incorrect_image_list,predicted_label_list,correct_label_list
            
            
def plot_misclassified_images(model, device, classes, testloader, num_of_images = 20, save_filename="misclassified"):
    model.eval()
    misclassified_cnt = 0
    fig = plt.figure(figsize=(10,9))
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)     # get the index of the max log-probability
        pred_marker = pred.eq(target.view_as(pred))   
        wrong_idx = (pred_marker == False).nonzero()  # get indices for wrong predictions
        for idx in wrong_idx:
          index = idx[0].item()
          title = "T:{}, P:{}".format(classes[target[index].item()], classes[pred[index][0].item()])
          #print(title)
          ax = fig.add_subplot(4, 5, misclassified_cnt+1, xticks=[], yticks=[])
          #ax.axis('off')
          ax.set_title(title)
          #plt.imshow(data[index].cpu().numpy().squeeze(), cmap='gray_r')
          imshow(data[index].cpu())
          
          misclassified_cnt += 1
          if(misclassified_cnt==num_of_images):
            break
        
        if(misclassified_cnt==num_of_images):
            break

    fig.savefig("./images/{}.png".format(save_filename))
    return            