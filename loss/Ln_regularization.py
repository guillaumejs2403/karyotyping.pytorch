import torch

def L1_Loss(model):
    regularization_loss = 0;
    for param in model.parameters():
        regularization_loss += torch.sum(torch.abs(param))
    return regularization_loss


def L2_Loss(model):
    regularization_loss = 0;
    for param in model.parameters():
        regularization_loss += torch.sum(torch.pow(param,2))
    return regularization_loss

def Ln_Loss(model,n):
    regularization_loss = 0
    if n%2==0:
        for param in model.parameters():
            regularization_loss += torch.sum(torch.pow(param,n))
    elif n%2 == 1:
        for param in model.parameters():
            regularization_loss += torch.sum(torch.pow(torch.abs(param),n))

    return regularization_loss

