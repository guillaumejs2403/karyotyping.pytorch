from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim

from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import shutil
import os
import math
import imageio

from networks import gcn
from python_scripts import learning_rate_decays as lrd, temp
import dataloader as dl
from loss.Ln_regularization import Ln_Loss
from test import run_test

import pdb

plt.ioff()
join = os.path.join
os.system('clear')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a karyotype network')


    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.

    # Epoch
    parser.add_argument(
        '--learning_rate',
        help=':D', dest = 'lr',
        default=0.00001, type=float, required = True)

    parser.add_argument(
        '--epochs',
        help=':D', dest = 'epochs',
        default=750, type=int, required = True)

    parser.add_argument(
        '--batch_size',
        help=':D', dest = 'batch_size',
        default=1, type=int, required = True)

    parser.add_argument(
        '--save_dir', help ='Set decay learning rate schedule',
        dest='save_dir',type = str, default = None, required = True)

    # learning rate schedule
    parser.add_argument(
        '--decay', help ='Set decay learning rate schedule',
        dest='decay',type = str2bool, default = False, required = False)

    parser.add_argument(
        '--schedule', help ='Set decay learning rate schedule',
        dest='schedule',type = int, default = None, required = False, nargs='+')

    parser.add_argument(
        '--divider', help ='Set decay learning rate schedule',
        dest='divider',type = float, default = None, required = False)

    parser.add_argument(
        '--each_ckpt', help ='Set decay learning rate schedule',
        dest='each',type = int, default = 0, required = False)

    parser.add_argument(
        '--Ln-lambda', help ='Set decay learning rate schedule',
        dest='ln_L',type = float, default = 1e-5, required = False)

    parser.add_argument(
        '--Ln-n', help ='Set decay learning rate schedule',
        dest='ln_n',type = int, default = 2, required = False)

    
    return parser.parse_args()

args = parse_args()


# ------------------------------------
#             Parameters
# ------------------------------------

print('Parameters:')
print(args)

batch_size = args.batch_size
lr = args.lr
momentum = 0.9
n_epochs = args.epochs
save_dir = args.save_dir 

ROOT_DIR_DATASET = '/media/SSD3/MFISH_Dataset/MFISH_split_normal'

# data augmentation online
data_aug = False

min_angle = -120
max_angle = 120

max_perturbation = 1.02
min_perturbation = 0.98

max_x = 0.1
max_y = 0.1

param = [min_angle,max_angle,max_x,max_y,min_perturbation,max_perturbation]

alpha = 1

n_reg = args.ln_n
L = args.ln_L

# Set to true if you want to save check points. if each % actual_epoch == 0 then make save ckpt
ckpt = True
if args.each <= 0 or args.each%1!=0:
    each = n_epochs//10
    print('WARNING: setting each checkpoint at iteration:', n_epochs,'%',each, '== 0 ')
else:
    each = args.each

decay = args.decay
schedule = args.schedule
divider = args.divider

if (schedule == None or divider == None or divider == 0) and decay:
    raise NameError('Need schedule od divider arguments')

if decay:
    learning_rate_decays = lrd.get_schedule(schedule,divider,lr)
    cont = 0

# ------------------------------------
#             
# ------------------------------------



dir_name = 'epochs_'+str(n_epochs)+'_lr_'+str(lr)+'_batchSize_'+str(batch_size) + '_L' + str(n_reg) + '_' + str(L)

if decay:
    dir_name = dir_name + '_withdecay_' + str(schedule)
    dir_name += '_divider_' + str(divider)

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists(join(save_dir, dir_name)):
    os.mkdir(join(save_dir, dir_name))

if not os.path.exists(join(save_dir, dir_name,'partial_segs')) and ckpt:
    os.mkdir(join(save_dir, dir_name,'partial_segs'))




# ------------------------------------
#             Ckpt Func
# ------------------------------------ 

def make_ckpt(network, root, dir_name, n_ckpt, loss):
    ckpt_name = 'ckpt_'+str(int(n_ckpt))
    if not os.path.exists(join(root,dir_name, ckpt_name)):
        os.mkdir(join(root,dir_name,ckpt_name))
    #else:
    #    shutil.rmtree(join(root,dir_name,ckpt_name))
    #    os.mkdir(join(root,dir_name,ckpt_name))
    torch.save(network.state_dict(), join(root,dir_name,ckpt_name,'model.pkl'))
    with open(join(root,dir_name,ckpt_name,'info_ckpt.txt'),'w') as info:
        info.write(str(n_ckpt) + '\n')
        info.write(str(loss[0])) # training loss
        info.write(str(loss[1])) # iter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------------
#             Load Model
# ------------------------------------

model = gcn.FCN(num_classes = 25)

# ------------------------------------
#             Load Dataset
# ------------------------------------

db_train = dl.MFISH_Dataset(root_dir = ROOT_DIR_DATASET, folder = 'train', add_augment = data_aug, rep = True)
train_dataloader =  data.DataLoader(db_train, batch_size = batch_size, shuffle = True, num_workers=1)

db_val = dl.MFISH_Dataset(root_dir = ROOT_DIR_DATASET, folder = 'val', add_augment = False, rep = True)
val_dataloader =  data.DataLoader(db_val, batch_size = 1, shuffle = False, num_workers=1)

# ------------------------------------
#             GPU Setings
# ------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------------------
#                  Loss
# ------------------------------------------


v = [1. for _ in range(25)]
v[0] = 1/10000
v[-1] = 10
weight = torch.Tensor(v)
criterion = nn.CrossEntropyLoss(weight = weight).to(device)
# cls_criterion =  nn.BCEWithLogitsLoss(weight = weight)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

# ------------------------------------------
#                Training
# ------------------------------------------

training_vec = []
val_vec = []

anan = False

print('Training is on!!!')

print('Number of parameters:', count_parameters(model))
timer = temp.Timer()
J = []

try:
    for epoch in range(n_epochs):

        temp.print_message(epoch,timer, n_epochs)

        running_loss =  0.0

        if decay:
            if len(learning_rate_decays)>cont:
                if epoch==(learning_rate_decays[cont][0]-1):
                    print('Setting new learning rate to', learning_rate_decays[cont][1])
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate_decays[cont][1])
                    cont += 1
        
        for i_batch, sample_batched in enumerate(train_dataloader):

            optimizer.zero_grad() # set gradient to 0
            image = sample_batched['image'].to(device,dtype = torch.float)
            label = sample_batched['label'].to(device,dtype = torch.long)

            result = model(image)
            
            if (epoch+1)%each == 0:

                ckpt_name = 'ckpt_'+str(int(epoch+1))
                segs = torch.argmax(result, dim = 1)

                if not os.path.exists(join(save_dir ,dir_name, ckpt_name)):
                    print(join(save_dir, dir_name, ckpt_name))
                    os.mkdir(join(save_dir, dir_name, ckpt_name))
                if not os.path.exists(join(save_dir, dir_name, ckpt_name,'partial_segs')):
                    print(join(save_dir, dir_name, ckpt_name,'partial_segs'))
                    os.mkdir(join(save_dir, dir_name, ckpt_name,'partial_segs'))
                for i in range(segs.shape[0]):
                    if np.random.rand() < 0.4:
                        temp_ = torch.squeeze(segs[i,:,:]).to('cpu').numpy().astype(np.uint8)*10
                        imageio.imwrite(join(save_dir, dir_name,ckpt_name,'partial_segs',str(i_batch) + '_' + str(i) + 't.png'),temp_)
            

            loss_c = criterion(result, label)
            loss_L = L*Ln_Loss(model,n_reg)
            loss = loss_c + loss_L
            running_loss += loss_c.item()*batch_size/92

            if math.isnan(running_loss):
                print('Nan found, making ckpt...')
                make_ckpt(model,save_dir,dir_name, epoch+1,[0,0])
                anan = True
                break

            loss.backward() # backward operation over the net
            optimizer.step() # some magic happends here

            if (i_batch+1) % 20 == 0 or (i_batch+1) == 92//batch_size:
                print('loss: {:.4f}'.format(running_loss),'| batch:', i_batch+1,'/', 92//batch_size ,'| epoch:', epoch+1,'/',n_epochs)


        if anan:
            with open(join(save_dir,dir_name,'nanlog.txt'),'w') as file:
                file.write('Nan achieved at epoch: '+str(epoch))
            break

        print('Epoch:',epoch+1, '|| Total loss training:', running_loss)
        training_vec.append(running_loss);


        
        # ==================================================== 
        # test on validation



        running_loss_val = 0.0

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(val_dataloader):

                optimizer.zero_grad() # set gradient to 0

                image = sample_batched['image'].to(device,dtype = torch.float)
                label = sample_batched['label'].to(device,dtype = torch.long)

                #pdb.set_trace() # puntico de pytlab
                result = model(image)
                if ckpt and (epoch+1)%each == 0:
                    # save images
                    # create segmentation
                    ckpt_name = 'ckpt_'+str(int(epoch+1))
                    segs = torch.argmax(result, dim = 1)
                    if not os.path.exists(join(save_dir ,dir_name, ckpt_name)):
                        os.mkdir(join(save_dir, dir_name, ckpt_name))
                        os.mkdir(join(save_dir, dir_name, ckpt_name,'partial_segs'))

                    for i in range(segs.shape[0]):
                        if np.random.rand() < 0.4:
                            temp_ = torch.squeeze(segs[i,:,:]).to('cpu').numpy().astype(np.uint8)*10
                            imageio.imwrite(join(save_dir, dir_name,ckpt_name,'partial_segs',str(i_batch) + '_' + str(i) + 'v.png'),temp_)

                loss = criterion(result, label) # gets the loss
                running_loss_val += loss.item()/46
            

        val_vec.append(running_loss_val)

        if ckpt and (epoch+1)%each == 0:
            print('Making check point')
            make_ckpt(model,save_dir,dir_name, epoch+1,[running_loss,running_loss_val])

        print('Epoch:',epoch+1, '|| Total loss validation:', running_loss_val)
        try:
            if epoch%1 == 0:
                j = run_test(model = model, dataloader = val_dataloader, path = join(save_dir, dir_name), device=device, save = False)
                J.append(j)
        except:
            J.append([0 for i in range(25)])

except KeyboardInterrupt:
    print('Training stopped by KeyboardInterrupt')
except:
    1+1
    print('=======================================')
    print('             Error found')
    print('=======================================')
    raise

print('Training finished!!')
print('Saving model')
torch.save(model.state_dict(),os.path.join(save_dir,dir_name,'model.pkl'))


fig = plt.figure(0)
plt.plot([i+1 for i in range(len(val_vec))],training_vec , 'b', label = 'Training Loss')
plt.plot([i+1 for i in range(len(val_vec))], val_vec, 'r', label = 'Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(join(save_dir,dir_name,'running_loss.png'))
plt.close(fig)

model.eval()
run_test(model = model, dataloader = val_dataloader, path = join(save_dir, dir_name), device = device)