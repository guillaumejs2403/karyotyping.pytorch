# extracted from https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183
import torch
import torch.nn as nn
import torch.nn.functional as F

class criterion():
    def __init__(self,smooth = 1, softmax = False):
        self.smooth = smooth
        self.softmax = softmax
        if softmax:
            self.soft_f = nn.Softmax(dim = 1)

    def dice_loss(self, pred, target):
        """This definition generalize to real valued pred and target vector.
        This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
        """ 
        if self.softmax:
            pred = self.soft_f(pred)
        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(tflat * iflat)
        B_sum = torch.sum(tflat * tflat)
    
        return 1 - ((2. * intersection + self.smooth) / (A_sum + B_sum + self.smooth))


def dice_coeff(pred, target, smooth = 1.):
    
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True,smooth = 1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        num = targets.size(0)  # Number of batches

        score = dice_coeff(probs, targets, self.smooth)
        score = 1 - score.sum() / num
        return score

def dice_loss(input, target, smooth = 1):
    
    loss = 0.
    for c in range(25):
           iflat = input[:, c ].view(-1)
           tflat = target[:, c].view(-1)
           intersection = (iflat * tflat).sum()
           
           w = 1#class_weights[c]
           loss += w*(1 - ((2. * intersection + smooth) /
                             (iflat.sum() + tflat.sum() + smooth)))
    return loss