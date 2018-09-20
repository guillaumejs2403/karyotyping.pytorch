from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

from scipy import ndimage
import matplotlib.pyplot as plt

# http://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/2


#padd = nn.ConstantPad2D(33,0)

# normalization required (all picture shouls be in 0 to 1 range before)
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
#rotation = transforms.RandomRotation(120)
#resize = transforms.RandomResizedCrop((645,517))
#ToTensor = transforms.ToTensor()

def forward(self,x):
        # realizar padding antes
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
    
        x = self.fc(x)
        return x

def get_model_1():
    
    model = models.resnet34(pretrained = True)
    model.fc = nn.Conv2d(512,25,1) # 512 cd 3x3 a 512
    # disable gradient for all layers
    for i in model.parameters():
        i.requires_grad = False

    # enable gradient for last layer
    model.fc.requires_grad = True
    model.forward = forward

    return model

# forward(model,x) da resultados