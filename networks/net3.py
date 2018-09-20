import torch
import torch.nn as nn
import torch.nn.functional as F

class simple_net(nn.Module):
    def __init__(self):
        super(simple_net,self).__init__()


        self.conv1_t1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, padding=1)
        self.conv2_t1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding=1)
        self.conv3_t1 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding=1)

        self.conv1_t2 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 5, padding=2)
        self.conv2_t2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 5, padding=2)
        self.conv3_t2 = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 5, padding=2)

        self.conv1_fc = nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding=1)
        self.conv2_fc = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding=1)
        self.conv3_fc = nn.Conv2d(in_channels = 32, out_channels = 25, kernel_size = 1, padding=0)

    def forward(self,x):

        x1 = F.relu(self.conv1_t1(x))
        x1 = F.relu(self.conv2_t1(x1))
        x1 = F.relu(self.conv3_t1(x1))

        x2 = F.relu(self.conv1_t2(x))
        x2 = F.relu(self.conv2_t2(x2))
        x2 = F.relu(self.conv3_t2(x2))

        x = torch.cat((x1,x2), dim = 1)
        x = F.relu(self.conv1_fc(x))
        x = F.relu(self.conv2_fc(x))
        x = self.conv3_fc(x)

        return x