from torch.nn import functional as F
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

class GradCAM:
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers 
    target_layers = list of convolution layer index as shown in summary
    """
    def __init__(self, model, candidate_layers=None):
        def save_fmaps(key):
          def forward_hook(module, input, output):
              self.fmap_pool[key] = output.detach()

          return forward_hook

        def save_grads(key):
          def backward_hook(module, grad_in, grad_out):
              self.grad_pool[key] = grad_out[0].detach()

          return backward_hook

        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.nll).to(self.device)
        print(one_hot.shape)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:] # HxW
        self.nll = self.model(image)
        #self.probs = F.softmax(self.logits, dim=1)
        return self.nll.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.nll.backward(gradient=one_hot, retain_graph=True)

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        # need to capture image size duign forward pass
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        # scale output between 0,1
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam

    
def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input    
    
def GRADCAM(images, labels, model, target_layers):
  model.eval()
  # map input to device
  #imgs =[]  
  #for img in images:
        #img = np.float32(cv2.resize(img, (224, 224))) / 255
        #input = preprocess_image(img)
        #input.required_grad = True
        #imgs.append(input)
  #images = imgs      
  images = torch.stack(images).to(model.device)
  # set up grad cam
  gcam = GradCAM(model, target_layers)
  # forward pass
  probs, ids = gcam.forward(images)
  # outputs agaist which to compute gradients
  ids_ = torch.LongTensor(labels).view(len(images),-1).to(model.device)
  # backward pass
  gcam.backward(ids=ids_)
  layers = []
  for i in range(len(target_layers)):
    target_layer = target_layers[i]
    print("Generating Grad-CAM @{}".format(target_layer))
    # Grad-CAM
    layers.append(gcam.generate(target_layer=target_layer))
  # remove hooks when done
  gcam.remove_hook()
  return layers, probs, ids

def process_for_grad_cam(incorrect_images_list,predicted_label_list,correct_label_list,transform_test):
    incorrect_images_disp  = incorrect_images_list[0:25]
    predicted_label_list = predicted_label_list[0:25]
    correct_label_list = correct_label_list[0:25]

    #dataiter = iter(testloader)
    #images, labels = dataiter.next()
    #img = images[15] 
    #img = img * 0.1
    #img = img + 0.5

    #img = images[10] / 2 + 0.5
    #img = img.numpy()
    #img=np.transpose(img, (1, 2, 0))
    #img=np.uint8(img*255)
    incorrect_images = []
    inc_img =[]
    correct_lb =[]
    predicted_lb = []
    for i in range(25) :
        img = incorrect_images_disp[i]
        img = img / 2 + 0.5
        img = img
        img=np.transpose(img, (1, 2, 0))
        img=np.uint8(img*4)

        #inc_img.append(img)
        #img = np.uint8(cv2.resize(img,(32,32)))
        img=transform_test(img)
        incorrect_images.append(img)
        predicted_lb.append(predicted_label_list[i])
        #print("The predicted is",predicted_label_list[i])
        correct_lb.append(correct_label_list[i])
        #print("The correct is",correct_label_list[i])
    #print(len(incorrect_images))
    #images_batch = torch.from_numpy(np.array(incorrect_images))
    return incorrect_images

def unnormalize(img):
    img = img.numpy().astype(dtype=np.float32)
    channel_means = [0.4914, 0.4822, 0.4465] 
    channel_stdevs = [0.2023, 0.1994, 0.2010]
    for i in range(img.shape[0]):
        img[i] = (img[i]*channel_stdevs[i])+channel_means[i]
  
    return np.transpose(img, (1,2,0))


def display_images(gcam_layers, disp_images,images, labels, target_layers, class_names, image_size, predicted,i,r,c):
    for j in range(len(images)):
        #img = np.uint8(255*unnormalize(images[j].view(image_size)))
        disp_img = disp_images[j]
        disp_img = disp_img / 2 + 0.5
        #img = img.cpu().numpy()
        disp_img=np.transpose(disp_img , (1, 2, 0))
        img = np.uint8(255*unnormalize(images[j].view(image_size)))
        if i==0:
          ax = plt.subplot(r, c, j+2)
          ax.text(0, 0.2, f"pred={class_names[predicted[j][0]]}\n[actual={class_names[labels[j]]}]", fontsize=14)
          plt.axis('off')
          plt.subplot(r, c, c+j+2)
          plt.imshow(disp_img, interpolation='bilinear')
          plt.axis('off')
          
        
        heatmap = 1-gcam_layers[i][j].cpu().numpy()[0] # reverse the color map
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.resize(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), (128,128))
        plt.subplot(r, c, (i+2)*c+j+2)
        plt.imshow(superimposed_img, interpolation='bilinear')
        
        plt.axis('off')



def PLOT(gcam_layers, disp_images,images, labels, target_layers, class_names, image_size, predicted):
    c = len(images[0:10])+1
    r = len(target_layers)+2
    fig = plt.figure(figsize=(32,14))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    ax = plt.subplot(r, c, 1)
    ax.text(0.3,-0.5, "INPUT", fontsize=14)
    plt.axis('off')
    for i in range(len(target_layers)):
        target_layer = target_layers[i]
        ax = plt.subplot(r, c, c*(i+1)+1)
        ax.text(0.3,-0.5, target_layer, fontsize=14)
        plt.axis('off')
        display_images(gcam_layers, disp_images,images, 
                     labels, target_layers, class_names, image_size,predicted,i,r,c)
          #display_images(gcam_layers, disp_images[10:20],images[10:20], 
                     #labels[0:10], target_layers, class_names, image_size,predicted[10:20],i,r,c)
          #display_images(gcam_layers, disp_images[20:25],images[20:25], 
                     #labels[0:10], target_layers, class_names, image_size,predicted[20:25],i,r,c)  
      
    plt.show()

def PLOTGRADCAM(gcam_layers, disp_images,images, labels, target_layers, class_names, image_size, predicted):
    PLOT(gcam_layers, disp_images[0:10],images[0:10], labels[0:10], target_layers, class_names, image_size, predicted[0:10])
    PLOT(gcam_layers, disp_images[10:20],images[10:20], labels[10:20], target_layers, class_names, image_size, predicted[10:20])
    PLOT(gcam_layers, disp_images[20:25],images[20:25], labels[20:25], target_layers, class_names, image_size, predicted[20:25])
    
    
    