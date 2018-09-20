import os
from PIL import Image
import numpy as np
from scipy import ndimage
import imageio

import torch
import torch.utils.data as data
from torchvision import transforms
# 

batch_size = 2
ROOT_DIR = '/media/SSD3/MFISH_Dataset/MFISH_split_normal'

# DATA AUGMENTATION

min_angle = -120
max_angle = 120

max_perturbation = 1.02
min_perturbation = 0.98

max_x = 0.1
max_y = 0.1

join = os.path.join

param = [min_angle,max_angle,max_x,max_y,min_perturbation,max_perturbation]

def get_random_square(im,gt,size):
    shape = im.shape
    if size>shape[0] or size>shape[1]:
        return None
    else:
        low = (np.random.randint(low = 0, high = shape[0]-2-size),np.random.randint(low = 0, high = shape[1]-2-size))
        return im[low[0]:low[0]+size,low[1]:low[1]+size], gt[low[0]:low[0]+size,low[1]:low[1]+size] 



def rotation_jitter(image,gt,min_angle = -120 , max_angle = 120):
    angle = np.random.randint(min_angle,max_angle)
    image_PIL = Image.fromarray(image)
    image_PIL = image_PIL.rotate(angle, resample = Image.BILINEAR, expand = False)
    gt_PIL = Image.fromarray(gt)
    gt_PIL = gt_PIL.rotate(angle, resample = Image.NEAREST, expand = False)
    return [np.asarray(image_PIL),np.asarray(gt_PIL)]

def color_jitter(image, min_perturbation = 0.98, max_perturbation = 1.02):
    ''' Changes the color of the image by a random factor between max_perturbation
        and min_perturbation. 
        Parameters:
            im: image
            max_perturbation: upper bound of the random factor
            min_perturbation: lower bound of the random factor'''
    rand = np.random.rand(image.shape[0],image.shape[1])*(max_perturbation - min_perturbation) + min_perturbation
    return (rand*image).astype(np.uint8)

def zoom_jitter(image,gt, max_x = 0.1, max_y = 0.1):
    ''' Get a resized image from the original one
        Parameters:
            image: image
            max_x: maximal fraction of zooming the image on the x-axis
            max_y: maximal fraction of zooming the image on the y-axis '''
    #dims = (image.shape[1],image.shape[0])
    dims = (512,512)
    y_min = int(np.random.rand()*max_y*dims[0])
    y_max = int(np.random.rand()*max_y*dims[0])
    x_min = int(np.random.rand()*max_x*dims[1])
    x_max = int(np.random.rand()*max_x*dims[1])
    image = Image.fromarray(image[y_min:-(y_max+1),x_min:-(x_max+1)])
    gt = Image.fromarray(gt[y_min:-(y_max+1),x_min:-(x_max+1)])
    image = np.asarray(image.resize(dims))
    gt = np.asarray(gt.resize(dims,Image.NEAREST))
    return [image,gt]

def flip(im,gt):
    if np.random.rand()<=50:
        im = np.flip(im,0)
        gt = np.flip(gt,0)
    if np.random.rand()<=50:
        im = np.flip(im,1)
        gt = np.flip(gt,1)
    return im,gt

def get_annotation_vol(target):
    dims = (target.shape[0],target.shape[1],25)
    annotations = np.zeros(dims, dtype = np.uint8)
    for i in range(25):
        if i != 24:
           annotations[:,:,i] = (target == i).astype(np.uint8)
        else:
            annotations[:,:,24] = (target == 255).astype(np.uint8)

    return annotations.transpose((2,0,1))

def get_correct_annotation(target, only_chr):
    dims = (target.shape[0],target.shape[1])
    annotations = np.zeros(dims)
    if only_chr:
        annotations = ((target>0).astype(np.float32))
    else:
        for i in range(25):
            if i != 24:
                annotations += i*((target==i).astype(np.float32))
            else:
                annotations += 24*((target==255).astype(np.float32))
    return annotations

def data_augmentation(aug, im, gt, param):
    if aug:
        im, gt = flip(im, gt)
        im, gt = rotation_jitter(im, gt, min_angle = -120, max_angle = 120)
    im, gt = get_random_square(im,gt, 512)
    return im, gt



class MFISH_Dataset():
    def __init__(self,root_dir,folder = 'train',transform=None,add_augment = False, totensor = True, rep = True, getdic = True, only_chr = False):
        self.add_augment = add_augment
        self.totensor = totensor
        self.transform = transform
        self.root_dir = os.path.join(root_dir,folder)
        self.image_dir = os.path.join(self.root_dir, folder + '2018')
        self.annotations_dir = os.path.join(self.root_dir, 'annotations')
        self.path_im = os.listdir(self.image_dir)
        self.path_im.sort()
        self.path_an = os.listdir(self.annotations_dir)
        self.path_an.sort()
        self.len = len(self.path_im)
        self.rep = rep
        self.getdic = getdic
        self.only_chr = only_chr

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        
        im = ndimage.imread(os.path.join(self.image_dir,self.path_im[idx])).astype(np.float32)
        gt = ndimage.imread(os.path.join(self.annotations_dir,self.path_an[idx])).astype(np.float32)
        gt = get_correct_annotation(gt, self.only_chr)
        im, gt = data_augmentation(self.add_augment, im, gt, 12)
        if self.rep:
            im = np.expand_dims(im, axis = 2)
            im = np.concatenate((im,im,im), axis = 2)

        im = im.transpose((2,0,1))

        if self.totensor:
            im = torch.tensor(im)#.as_tensor(im,dtype = torch.float)
            gt = torch.tensor(gt)#.as_tensor(im,dtype = torch.float)

        if self.getdic:
            return {'image':im,'label':gt}
        else:
            return im, gt
        
if __name__ == '__main__':
    os.system('clear')
    db = MFISH_Dataset(root_dir = ROOT_DIR, folder = 'train', add_augment = True, totensor = False, getdic = False, only_chr = True)
    for i in range(db.__len__()):
        im,gt  = db.__getitem__(i)
        imageio.imwrite('data/im/im_'+str(i)+'.png',im.astype(np.uint8).transpose(1,2,0))
        imageio.imwrite('data/gt/gt_'+str(i)+'.png',gt.astype(np.uint8)*255)