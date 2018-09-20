import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class chr_net(nn.Module):
    def __init__(self,block=16,inter = 50):
        super(chr_net,self).__init__()

        self.conv1 = conv_block_2(1,block,block)
        self.conv2 = conv_block_2(block,2*block,2*block)
        self.conv3 = conv_block_2(2*block,4*block,4*block)

        self.tanh = nn.Tanh()
        self.central = nn.Conv2d(in_channels = 4*block, out_channels = 4*block, kernel_size = 1)

        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.deconv3 = deconv_block_2(8*block,2*block,4*block)
        self.deconv2 = deconv_block_2(4*block,block,2*block)
        self.deconv1 = deconv_block_2(2*block,inter,inter)

        self.fc1 = nn.Conv2d(in_channels = inter, out_channels = inter, kernel_size = 1)
        self.fc2 = nn.Conv2d(in_channels = inter, out_channels = 25, kernel_size = 1)

    def forward(self,x):

        d1 = self.conv1(x) # 1x644x516 -> 32x642x514 -> 32x640x512
        d2 = F.avg_pool2d(d1, kernel_size = 2) # 32x640x512 -> 32x320x256
        d2 = self.conv2(d2) # 32x320x256 -> 64x316x252
        d3 = F.avg_pool2d(d2,kernel_size = 2) # 64x316x252 -> 64x158x126
        d3 = self.conv3(d3) # 64x158x126 -> 128x154x122
        c = F.avg_pool2d(d3,kernel_size = 2) # 128x154x122 -> 128x77x61
        c = self.tanh(c)
        c = self.central(c) # 128x77x61 -> 128x77x61
        c = self.tanh(c)
        up3 = self.up(c) # 128x77x61 -> 128x154x122
        up3 = torch.cat((up3,d3), dim = 1) # 128x154x122 + 128x154x122 -> 256x154x122
        up3 = self.deconv3(up3) # 256x154x122 -> 64x158x126
        up2 = self.up(up3) # 64x158x126 -> 64x316x252
        up2 = torch.cat((up2,d2), dim = 1) # 64x316x252 + 64x316x252 -> 128x316x252
        up2 = self.deconv2(up2) # 128x316x252 -> 32x320x256
        up1 = self.up(up2) # 32x320x256 -> 32x640x512
        up1 = torch.cat((up1,d1), dim = 1) # 32x640x512 + 32x640x512 -> 64x640x512
        up1 = self.deconv1(up1) # 64x640x512 -> 50x644x516

        x = F.relu(self.fc1(up1))
        x = F.relu(self.fc2(x))
        
        return x

class chr_net_v2(nn.Module):
    def __init__(self,block=16,inter = 50):
        super(chr_net_v2,self).__init__()

        self.conv1 = conv_block_2(1,block,block)
        self.conv2 = conv_block_2(block,2*block,2*block)
        self.conv3 = conv_block_2(2*block,4*block,4*block)

        self.tanh = nn.Tanh()
        self.central = nn.Conv2d(in_channels = 4*block, out_channels = 4*block, kernel_size = 1)

        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear')
        self.deconv3 = deconv_block_2(8*block,2*block,4*block)
        self.deconv2 = deconv_block_2(4*block,block,2*block)
        self.deconv1 = deconv_block_2(2*block,inter,inter)

        self.fc1 = nn.Conv2d(in_channels = inter, out_channels = inter, kernel_size = 1)
        self.fc2 = nn.Conv2d(in_channels = inter, out_channels = 25, kernel_size = 1)

    def forward(self,x):

        d1 = self.conv1(x) # 1x644x516 -> 32x642x514 -> 32x640x512
        d2 = F.max_pool2d(d1, kernel_size = 2) # 32x640x512 -> 32x320x256
        d2 = self.conv2(d2) # 32x320x256 -> 64x316x252
        d3 = F.max_pool2d(d2,kernel_size = 2) # 64x316x252 -> 64x158x126
        d3 = self.conv3(d3) # 64x158x126 -> 128x154x122
        c = F.max_pool2d(d3,kernel_size = 2) # 128x154x122 -> 128x77x61
        c = self.central(c) # 128x77x61 -> 128x77x61
        c = F.relu(c)
        up3 = self.up(c) # 128x77x61 -> 128x154x122
        up3 = torch.cat((up3,d3), dim = 1) # 128x154x122 + 128x154x122 -> 256x154x122
        up3 = self.deconv3(up3) # 256x154x122 -> 64x158x126
        up2 = self.up(up3) # 64x158x126 -> 64x316x252
        up2 = torch.cat((up2,d2), dim = 1) # 64x316x252 + 64x316x252 -> 128x316x252
        up2 = self.deconv2(up2) # 128x316x252 -> 32x320x256
        up1 = self.up(up2) # 32x320x256 -> 32x640x512
        up1 = torch.cat((up1,d1), dim = 1) # 32x640x512 + 32x640x512 -> 64x640x512
        up1 = self.deconv1(up1) # 64x640x512 -> 50x644x516

        x = F.relu(self.fc1(up1))
        x = F.relu(self.fc2(x))
        
        return x


class conv_block_3(nn.Module):
    def __init__(self, channels_in, channels_out, n_features, padding = 0,kernel_size = 3):
        super(conv_block_3,self).__init__()
        self.con1 = nn.Conv2d(in_channels = channels_in, out_channels = n_features,kernel_size = kernel_size, padding = padding)
        self.con2 = nn.Conv2d(in_channels = n_features, out_channels = n_features,kernel_size = kernel_size, padding = padding)
        self.con3 = nn.Conv2d(in_channels = n_features, out_channels = channels_out,kernel_size = kernel_size, padding = padding)

    def forward(self,x):
        x = F.relu(self.con1(x))
        x = F.relu(self.con2(x))
        x = F.relu(self.con3(x))
        return x

class conv_block_2(nn.Module):
    def __init__(self, channels_in, channels_out, n_features, padding = 0,kernel_size = 3):
        super(conv_block_2,self).__init__()
        self.con1 = nn.Conv2d(in_channels = channels_in, out_channels = n_features,kernel_size = kernel_size, padding = padding)
        self.con2 = nn.Conv2d(in_channels = n_features, out_channels = channels_out,kernel_size = kernel_size, padding = padding)

    def forward(self,x):
        x = F.relu(self.con1(x))
        x = F.relu(self.con2(x))
        return x

class deconv_block_3(nn.Module):
    def __init__(self, channels_in, channels_out, n_features, padding = 0, kernel_size = 3):
        super(deconv_block_3,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels = channels_in, out_channels = n_features, kernel_size = kernel_size, padding = padding)
        self.deconv2 = nn.ConvTranspose2d(in_channels = n_features, out_channels = n_features, kernel_size = kernel_size, padding = padding)
        self.deconv3 = nn.ConvTranspose2d(in_channels = n_features, out_channels = channels_out, kernel_size = kernel_size, padding = padding)

    def forward(self,x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        return x

class deconv_block_2(nn.Module):
    def __init__(self, channels_in, channels_out, n_features, padding = 0, kernel_size = 3):
        super(deconv_block_2,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(in_channels = channels_in, out_channels = n_features, kernel_size = kernel_size, padding = padding)
        self.deconv2 = nn.ConvTranspose2d(in_channels = n_features, out_channels = channels_out, kernel_size = kernel_size, padding = padding)

    def forward(self,x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        return x

if __name__ == '__main__':
    model = chr_net()
    input =  torch.rand(4,1,644,516)
    output = model(input)

    #cb2 = conv_block_2(1,1,1)
    #output = cb2(input)
    print('output:',output.shape)
