import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import models, transforms

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


class block(nn.Module):
    def __init__(self, features = [1, 8, 16]):
        self.conv1 = nn.Conv2d(features[0], features[1], kernel_size = 1)
        


class Net(nn.Module):
    def __init__(self, backbone = 'resnet18' ,bb_grad = False):
        super(Net,self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        if not bb_grad:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Head convolutional network
        self.head = nn.Conv2d()