# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:59:26 2024

@author: Mateo-drr
"""
import torch.nn as nn
import torch
import torch.optim as optim

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
###########################################################################
#FUNCTIONS
###########################################################################
def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_accuracy(logits, y, notLogit=False):
    if notLogit: #used for actis
        _, base_predicted = torch.max(logits, 1)
        base_loss = (base_predicted == y.squeeze(1)).sum() #giving error because y had shape (samples,1) while the other had (samples,)
        return base_loss/logits.size(0)
    else:
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y):
    scale = torch.tensor(1.).cuda().requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

def prepDat(data):
    img, lbl = data
    img, lbl = img.to(device)[:,:-1,:,:], lbl.to(device).unsqueeze(1).float() #remove blue channel and set lbls as floats
    img = img[:,:, ::2, ::2] #same as source code use 14*14 instead of 28*28
    return img,lbl

def fwdPass(model,criterion,img,lbl,totloss,op='ERM'):
    # Forward pass
    outputs = model(img)    
    if op == 'ERM':
        loss = criterion(outputs, lbl)
        totloss += loss.item() * img.size(0)  # Multiply by batch size
    elif op == 'IRM':
        loss = mean_nll(outputs, lbl)
        totloss += loss.mean().item() * img.size(0)
    return totloss, loss, outputs

def initModel(lr,model,tmax):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax,eta_min=0.00001,verbose=True)
    return model,optimizer,scheduler

def wblog(wandb,ce,ce2,cet,acc,acc2,acct,sfb,pCI,pStability):
    wandb.log({"Train ce1": ce,
           "Train ce2": ce2,
           "Test ce": cet,
           "Train acc1": acc.mean(),
           "Train acc2": acc2.mean(),
           "Test acc": acct.mean(),
           "SFB loss": sfb,
           "pCI": pCI,
           "pS":pStability})

def wbinit(wandb,cf,name):
    wandb.init(
        # set the wandb project where this run will be logged
        name=name,
        project="StableFeatureBoosting",
    
        # track hyperparameters and run metadata
        config=cf
    )
###############################################################################
#ACTIS
###############################################################################
def DiscreteConditionalExpecationTest(x, y, z):
    '''
    Parameters
    ----------
    x : tensor
        f beta ie the base prediction / stable prediction.
    y : tensor
        f eta ie the enviroment prediction / unstable prediction.
    z : tensor
        labels that the model needs to predict.

    Returns
    -------
    Conditional independence value
        Value to enforce conditional independence of x and y given z.
        Complete formula: E[x*(y-E[y|z])]
    '''
    n, _ = x.shape #get the number of samples
    if len(z.shape) > 1: #decide wether to squeeze the tensor from [b,1] to [b]
      temp_z = z[:,0] 
    else:
      temp_z = z
      
    labels_in_batch_sorted, indices = torch.sort(temp_z) #sort the labels, putting 0 first then 1s
    unique_ixs = 1 + (labels_in_batch_sorted[1:] - labels_in_batch_sorted[:-1]).nonzero() #find the index where labels go from 0 to 1
    unique_ixs = [0] + unique_ixs.flatten().cpu().numpy().tolist() + [len(temp_z)] # turn to a list [0, index of change, b]
    
    estimate = 0 
    for j in range(len(unique_ixs)-1): #calculate the estimation of the labels combining eta and beta for each class type
      current_class_indices = unique_ixs[j], unique_ixs[j + 1] #getting the number of samples with label 0 and label 1
      count = current_class_indices[1] - current_class_indices[0] #same
      if count < 2: 
        continue
      curr_class_slice = slice(*current_class_indices) #creates an slice veriable
      curr_class_indices = indices[curr_class_slice].sort()[0] #get only the indices of the current class
    
      #E[y|z]
      y_cond_z = torch.mean(y[curr_class_indices, :], dim=0, keepdim=True) #take the mean predicted label in each 8 channels
      #yy = y - E[y|z]
      yy = y[curr_class_indices, :] - y_cond_z
      #sum(x*yy) ---- x is the beta/stable predictions and they are multiplied with the shifted eta/unstable predictions
      estimate += torch.sum(x[curr_class_indices, :] * (yy), dim=0) 
    
    return estimate/n #E[]