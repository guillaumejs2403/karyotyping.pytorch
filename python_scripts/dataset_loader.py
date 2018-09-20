import os
from PIL import Image
import numpy as np
from scipy import ndimage, misc

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
    dims = (644,516)
    #dims = (512,512)
    y_min = int(np.random.rand()*max_y*dims[0])
    y_max = int(np.random.rand()*max_y*dims[0])
    x_min = int(np.random.rand()*max_x*dims[1])
    x_max = int(np.random.rand()*max_x*dims[1])
    image = Image.fromarray(image[y_min:-(y_max+1),x_min:-(x_max+1)])
    gt = Image.fromarray(gt[y_min:-(y_max+1),x_min:-(x_max+1)])
    image = np.asarray(image.resize(dims))
    gt = np.asarray(gt.resize(dims,Image.NEAREST))
    #gt = np.asarray(gt.resize((388,388),Image.NEAREST))
    return [image,gt]

def flip(im,gt):
    if np.random.rand()<=50:
        im = np.flip(im,0)
        gt = np.flip(gt,0)
    if np.random.rand()<=50:
        im = np.flip(im,1)
        gt = np.flip(gt,1)
    return [im,gt]

def get_annotation_vol(target):
    dims = (target.shape[0],target.shape[1],25)
    annotations = np.zeros(dims, dtype = np.uint8)
    for i in range(25):
        if i != 24:
           annotations[:,:,i] = (target == i).astype(np.uint8)
        else:
            annotations[:,:,24] = (target == 255).astype(np.uint8)

    return annotations.transpose((2,0,1))

def get_correct_annotation(target):
    dims = (target.shape[0],target.shape[1])
    annotations = np.zeros(dims)
    for i in range(25):
        if i != 24:
            annotations += i*((target==i).astype(np.float32))
        else:
            annotations += 24*((target==255).astype(np.float32))
        #print(np.amin(annotations),np.amax(annotations))
    return annotations





class MFISH_Dataset():
    def __init__(self,root_dir,folder = 'train',transform=None,add_augment = False, param = None, vol = False):
        self.add_augment = add_augment
        self.param = param
        self.transform = transform
        self.root_dir = os.path.join(root_dir,folder)
        self.image_dir = os.path.join(self.root_dir, folder + '2018')
        self.annotations_dir = os.path.join(self.root_dir, 'annotations')
        self.path_im = os.listdir(self.image_dir)
        self.path_im.sort()
        self.path_an = os.listdir(self.annotations_dir)
        self.path_an.sort()
        self.len = len(self.path_im)
        self.vol = vol

    def __len__(self):
        return self.len

    def __getitem__(self,idx):
        param = self.param
        im = ndimage.imread(os.path.join(self.image_dir,self.path_im[idx]))
        gt = ndimage.imread(os.path.join(self.annotations_dir,self.path_an[idx]))
        
        im = im[:-1,:-1].astype(np.float32)
        gt = gt[:-1,:-1].astype(np.float32)
        if self.add_augment:

            # [min_angle,max_angle,max_x,max_y,min_perturbation,max_perturbation]
            im, gt = flip(im,gt)
            im, gt = rotation_jitter(im,gt, param[0],param[1])
            im, gt = zoom_jitter(im,gt,param[2],param[3])
            # im = color_jitter(im,param[4],param[5])
            
        else:

            im = Image.fromarray(im)
            gt = Image.fromarray(gt)
            im = np.asarray(im.resize((644,516)))
            gt = np.asarray(gt.resize((644,516),Image.NEAREST))     

        #misc.imsave('gt.png',gt*10)
        #misc.imsave('im.png',im)
        im = np.expand_dims(im,axis = 2)
        im = (im).transpose((2,0,1))
        #im = np.concatenate((im,im,im), axis = 2)
        if self.vol:
            gt = get_annotation_vol(gt)
        else:
            gt = get_correct_annotation(gt) #get_correct_annotation(np.expand_dims(gt, axis = 2))
            #gt = gt.transpose((2,0,1))
        
        #gt = np.expand_dims(gt,axis = 2).transpose((2,0,1))
        

        im = torch.tensor(im)#.as_tensor(im,dtype = torch.float)
        gt = torch.tensor(gt)#.as_tensor(im,dtype = torch.float)

        return {'image':im,'label':gt}

if __name__ == '__main__':
    db_train = MFISH_Dataset(root_dir = ROOT_DIR, folder = 'val', add_augment = False, param = param, vol = False)
    train_dataloader =  data.DataLoader(db_train, batch_size = 1, shuffle = False, num_workers=1)
    folder = 'ims'
    
    for i_batch, sample_batched in enumerate(train_dataloader):
        print(i_batch)
        