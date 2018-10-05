import torch
import numpy as np
import os

import dataloader as dl
import torch.utils.data as data
import scipy.misc as m
from networks import PSPnet

join = os.path.join

def multi_idx_jaccard(pred,label, categories = 25, device = None):
    
    val = torch.zeros(categories)
    # Segmentation 
    label = torch.squeeze(label).to('cpu').numpy()
    segmentation = torch.argmax(pred, dim = 1)
    segmentation = torch.squeeze(segmentation).to('cpu').numpy()
    for i in range(categories):
        inter = np.sum(np.logical_and(segmentation== i, label == i))
        union = np.sum(np.logical_or(segmentation == i, label == i))
        val[i] = inter/union

    return [val,segmentation]


# ------------------------------------
#              Run test
# ------------------------------------


def run_test(model, dataloader,path, device, save = True, categories = 25):


    if not os.path.exists(join(path,'results')):
        os.mkdir(join(path,'results'))

    for i_batch, sample_batched in enumerate(dataloader):
        inputs = sample_batched['image'].to(device,dtype = torch.float)
        labels = sample_batched['label'].to(device,dtype = torch.float)
        
        pred = model(inputs)

        n += labels.shape[0]
        if i_batch == 0:
            jaccard, segmentation = multi_idx_jaccard(pred,labels)
        else:
            result = multi_idx_jaccard(pred,labels)
            jaccard += result[0]
            segmentation = result[1]

        ''' Saving '''
        if save:
            for i in range(labels.shape[0]):
                temp = segmentation.astype(np.uint8)*10
                m.imsave(join(path,'results',str(s)+'.png'),temp)
                s += 1


    jaccard /= n
    print('Jaccard:\n')
    ss = ''
    for i in range(categories):
        ss += '| Category {}: {:.3f} |'.format(i,jaccard[i])
    print(ss)
    return jaccard


if __name__ == '__main__':

    os.system('clear')

    # ------------------------------------
    #             Load Model
    # ------------------------------------

    print('Loading Model')
    model = PSPnet.PSPNet(n_classes = 25,psp_size=512, deep_features_size=256, backend='resnet34')
    model.load_state_dict(torch.load(join(DATA_MODEL,'model.pkl')))
    model.to(device)
    model.eval() # Set model to test mode

    # ------------------------------------
    #            GPU Settings
    # ------------------------------------

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ROOT_DIR_DATASET = '/media/SSD3/MFISH_Dataset/MFISH_split_normal'
    DATA_MODEL = '/media/user_home4/gjeanneret/karyotyping.pytorch/psp_net-resnet34-l1r/epochs_300_lr_5e-05_batchSize_5/ckpt_50'

    # ------------------------------------
    #              Dataset  
    # ------------------------------------

    db_val = dl.MFISH_Dataset(root_dir = ROOT_DIR_DATASET, folder = 'val', add_augment = False, rep = True)
    val_dataloader =  data.DataLoader(db_val, batch_size = 1, shuffle = False, num_workers=1)

    J = run_test(model = model, dataloader = val_dataloader, path = DATA_MODEL)
