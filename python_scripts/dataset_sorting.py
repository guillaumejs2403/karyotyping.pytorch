import os
import numpy as np
from scipy import ndimage
import scipy.misc as misc
from PIL import Image

# -------------------------------------------------------------------- #
#                            Parameters                                #
# -------------------------------------------------------------------- #

PATH = '/media/SSD3/MFISH_Dataset/MFISH_original'                    # path to the original dataset
NAME_OUT = 'MFISH_split_normal'                                        # name of the new folder
PATH_SORT = '/media/SSD3/MFISH_Dataset'                              # folder which will contain the processed dataset

train_partition = 50
test_partition = 25
val_partition = 25

rotation = False
rotation_vect = [30,60,90,120,150,180]                                 # vector containing the retoation image

reflection = False

jitter = False
n_jitter = 10                                                          # number of jitters per image
max_g = 1.5                                                            # max change in grayscale values
min_g = 0.5                                                            # min change in grayscale values
max_z = 0.25                                                           # max scaling value


# This functions is should return an image given a list containing 2 pictures ([picture1, picture2])

def best(im_vec):
    if np.std(im_vec[0],keepdims=False)>np.std(im_vec[1],keepdims=False):
        return im_vec[0]
    else:
        return im_vec[1]

def max_im(im_vec):
    return np.amax(im_vec, axis = 0)


# -------------------------------------------------------------------- #
#                                end                                   #
# -------------------------------------------------------------------- #
PATH_SORT = os.path.join(PATH_SORT, NAME_OUT)

os.system('rm -r '+PATH_SORT)


if not os.path.exists(os.path.join(PATH_SORT)):
    os.mkdir(os.path.join(PATH_SORT))
    os.mkdir(os.path.join(PATH_SORT,'train'))
    os.mkdir(os.path.join(PATH_SORT,'train','annotations'))
    os.mkdir(os.path.join(PATH_SORT,'train','train2018'))
    os.mkdir(os.path.join(PATH_SORT,'val'))
    os.mkdir(os.path.join(PATH_SORT,'val','annotations'))
    os.mkdir(os.path.join(PATH_SORT,'val','val2018'))
    os.mkdir(os.path.join(PATH_SORT,'test'))
    os.mkdir(os.path.join(PATH_SORT,'test','annotations'))
    os.mkdir(os.path.join(PATH_SORT,'test','test2018'))


class db():
    def __init__(self,PATH_in,PATH_out,train_p=50,val_p=25,test_p=25):
        folders = os.listdir(PATH)
        folders.remove('README.TXT')
        temp = []
        self.main_list = []
        for folder in folders:
            folders_in = os.listdir(os.path.join(PATH_in,folder))
            ID_main = folder[0:3]
            #temp.append(folder[0:3])
            images_ID = []
            chain = []
            # take the images within the folder
            #print(folder)
            path_to_main = os.listdir(os.path.join(PATH_in,folder))
            path_to_main.sort()
            try:
                if path_to_main.index('Case Info')!=999999:
                    path_to_main.remove('Case Info')
            except ValueError:
                print(path_to_main)
            
            self.main_list.append([len(path_to_main),folder,path_to_main])
        self.main_list.sort(reverse = True)
        self.PATH_in = PATH_in
        self.PATH_out = PATH_out
        self.train = {'len':0}
        self.val = {'len':0}
        self.test = {'len':0}
        self.max_path = []
        self.train_p = train_p
        self.test_p = test_p
        self.val_p = val_p
    def sort(self,function,rotation,reflection,jitter,rotation_vect, n_jitter,max_g ,min_g ,max_z):
        t = 0
        now = 0
        print(rotation,reflection,jitter)
        for n,_,_ in self.main_list:
            t += n
        for n,folder,main in self.main_list:
            now += n
            # get the partition per folder
            IDs = []
            tupla = [] # [label(ID+ID_temp+.png),max,ground_truth]
            for image in main:
                ID_temp = image[0:7]
                im = ndimage.imread(os.path.join(self.PATH_in, folder, image))
                if ID_temp not in IDs:
                    IDs.append(ID_temp)
                    tupla.append([ID_temp,np.zeros(im.shape),np.zeros(im.shape)])
                    if 'K.png' in image:
                        tupla[-1][2] = im
                    else:
                        tupla[-1][1] = im
                else:
                    idx = IDs.index(ID_temp)
                    if 'K.png' in image:
                        tupla[idx][2] = im
                    else:
                        # -----------------------------------#
                        #    here it is done the operation   #
                        # -----------------------------------#
                        tupla[idx][1] = function( [tupla[idx][1],im])

            # adds to train, val or test every image
            if self.train['len']/self.train_p <= self.val['len']/self.val_p and self.train['len']/self.train_p < self.test['len']/self.test_p:
                # saves image and anotation
                self.train['len'] += save_tuple(tupla,'train',self.PATH_out,rotation,reflection,jitter,rotation_vect, n_jitter ,max_g ,min_g ,max_z )
                
            elif self.test['len']/self.test_p <= self.train['len']/self.train_p and self.test['len']/self.test_p <= self.val['len']/self.val_p:
                # saves image and anotation
                self.test['len'] += save_tuple(tupla,'test',self.PATH_out)
                
            elif self.val['len']/self.val_p < self.test['len']/self.test_p and self.val['len']/self.val_p < self.train['len']/self.train_p:
                # saves image and anotation
                self.val['len'] += save_tuple(tupla,'val',self.PATH_out)
            print('Images processed:', now, '/', t)
                

def save_tuple(tupla, folder, PATH_out, rotation=False, reflection=False, jitter = False, rotation_vect = [30, 60, 90], n_jitter = 10, max_g = 1.1,min_g = 0.9,max_z = 0.05):
    # check if there's no K's
    cont = 0
    for ID, image, ground_truth in tupla:
        if np.sum(ground_truth)!=0:
            name = ID+'.png'
            fullfile_im = os.path.join(PATH_out,folder,folder+'2018',name)
            fullfile_gt = os.path.join(PATH_out,folder,'annotations',name)
            misc.imsave(fullfile_im,image)
            misc.imsave(fullfile_gt,ground_truth)
            if rotation:
                for idx, angle in enumerate(rotation_vect):
                    name = ID+'r'+str(idx)+'.png'
                    fullfile_im = os.path.join(PATH_out,folder,folder+'2018',name)
                    fullfile_gt = os.path.join(PATH_out,folder,'annotations',name)
                    misc.imsave(fullfile_im,ndimage.interpolation.rotate(image, angle = angle, reshape = False))
                    misc.imsave(fullfile_gt,ndimage.interpolation.rotate(ground_truth, angle = angle, reshape = False, mode = 'nearest'))
            if reflection:
                tupla = [(image.shape,True,False),(image.shape,False,True),(image.shape,True,True)]
                for idx,t in enumerate(tupla):
                    name = ID + 'x'+str(idx)+'.png'
                    fullfile_im = os.path.join(PATH_out,folder,folder+'2018',name)
                    fullfile_gt = os.path.join(PATH_out,folder,'annotations',name)
                    misc.imsave(fullfile_im,ndimage.geometric_transform(image,reflex, extra_arguments = t))
                    misc.imsave(fullfile_gt,ndimage.geometric_transform(ground_truth,reflex, extra_arguments = t))
            if jitter:
                im_list, gr_list = jitter_fun(image,ground_truth,n_jitter,max_g,min_g,max_z)
                for idx, im in enumerate(im_list):
                    name = ID + 'j'+str(idx)+'.png'
                    fullfile_im = os.path.join(PATH_out,folder,folder+'2018',name)
                    fullfile_gt = os.path.join(PATH_out,folder,'annotations',name)
                    misc.imsave(fullfile_im,im)
                    misc.imsave(fullfile_gt,gr_list[idx])
            cont += 1
    return cont

def reflex(output_coords,c_max,vert,horz):
    if vert and not horz:
        return(c_max[0]-output_coords[0]+1,output_coords[1])
    elif horz and not vert:
        return(output_coords[0],c_max[1]-output_coords[1]+1)
    elif vert and horz:
        return(c_max[0]-output_coords[0]+1,c_max[1]-output_coords[1]+1)

def jitter_fun(im, gt, n_jitter = 10, max_g = 1.1, min_g = 0.9, max_z = 0.05):
    to_return_im = []
    to_return_gt = []
    for n in range(n_jitter):
        im_to_copy = gray_reshape(im,max_g,min_g)
        im_to_copy, gt_to_copy = jit(im_to_copy,gt,max_z)
        to_return_im.append(im_to_copy)
        to_return_gt.append(gt_to_copy)

    return [to_return_im,to_return_gt]

def gray_reshape(im,max_g,min_g):
    r_n = (max_g - min_g )*np.random.rand()+min_g
    return im*r_n//1

def jit(im, gt, max_z):
    dims = im.shape
    y_min = int(np.random.rand()*max_z*dims[0])
    y_max = int(np.random.rand()*max_z*dims[0])
    x_min = int(np.random.rand()*max_z*dims[1])
    x_max = int(np.random.rand()*max_z*dims[1])
    im = Image.fromarray(im[y_min:-(y_max+1),x_min:-(x_max+1)])
    gt = Image.fromarray(gt[y_min:-(y_max+1),x_min:-(x_max+1)])
    im = im.resize(dims)
    gt = gt.resize(dims,Image.NEAREST)
    return [im,gt]
    


cariotipo = db(PATH,PATH_SORT,train_partition,val_partition,test_partition)
cariotipo.sort(best,rotation,reflection,jitter,rotation_vect,n_jitter,max_g ,min_g,max_z)
