import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models



"""class MyVGG16(torch.nn.Module):
    def __init__(self, device):
        super(MyVGG16, self).__init__()
        self.device = device  # Store device as an instance variable
        self.model = models.vgg16(weights='IMAGENET1K_FEATURES')
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        self.shape = 25088 # the length of the feature vector

    def extract_features(self, image):
        transform = transforms.Compose([transforms.Normalize(mean=[0.48235, 0.45882, 0.40784], 
                                    std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
        image = transform(image)
        image = image.to(self.device)  # Move image to GPU

        # Feed the image into the model for feature extraction
        with torch.no_grad():
            feature = self.model.features(image)
            feature = torch.flatten(feature, start_dim=1)

        # Return features to numpy array
        return feature.cpu().detach().numpy()
"""
class MyResnet50(torch.nn.Module):
    def __init__(self, device):
       
       
        super(MyResnet50, self).__init__()
        self.device = device  # Set the device parameter to self.device

        self.model = models.resnet50(pretrained=True)
        # Remove the classification head
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.eval()
        self.model = self.model.to(self.device)
        self.shape = 2048 

    def extract_features(self, image):
        transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                            std=[0.229, 0.224, 0.225])])
        image = transform(image)
        image = image.to(self.device)
        # Pass the image through the Resnet50 model and get the feature maps of the input image
        with torch.no_grad():
            feature = self.model(image)
            feature = torch.flatten(feature, start_dim=1)
        # Return features to numpy array
        return feature.squeeze().tolist()