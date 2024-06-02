# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:28:05 2024

@author: Mateo-drr
"""

import torch
import torch.nn as nn
import torch.optim as optim
import CMNIST as dl
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils import mean_accuracy,prepDat,fwdPass,penalty, mean_nll, initModel, wbinit
from models import ERMIRM, Classifier

import torch.nn.init as init
from IRM import runIRM
from ERM import runERM
from SFB import runSFB
from ACTIR import runACTIR, initAdapInv, regressionLoss
import wandb

#PARAMS
run=9
cf = {
    "num_epochs": 1000,
    "lr": 0.0004898536566546834,
    "hsize": 390,
    "path": 'D:/MachineLearning/datasets',
    "pth": 'D:/Universidades/Trento/3S/AML/',
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "bsize": 25000,
    "img_size": 2 * 14 * 14,
    "pred_size": 1,
    "printFreq": 1,
    #IRM
    "l2_regularizer_weight": 0.00110794568,
    "penlt_weight": 91257.18613115903,
    "penalty_anneal_iters": 190,
    "base_irm_weight": 0.1,
    #ACTIS
    "num_env": 2,
    #SFB
    "lambdaC": 0,
    #OTHER
    "sch": True,
    "wb": True,
    "seed": 0,
    "lock": False, # True to not change lr
    "rescale": True,
    "clip": False,
    "gclip": 5,
    "fixedSeed":False
}

if cf['fixedSeed']:
    torch.manual_seed(cf['seed'])
    

torch.set_num_threads(8)
torch.set_num_interop_threads(8)
torch.backends.cudnn.benchmark=True

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(), # Convert PIL image to tensor
])  

# Create datasets for training and test environments
train_dataset_1 = dl.ColoredMNIST(root=cf['path'], env='train1', transform=transform)
train_dataset_2 = dl.ColoredMNIST(root=cf['path'], env='train2', transform=transform)
test_dataset = dl.ColoredMNIST(root=cf['path'], env='test', transform=transform)

# Create data loaders
train_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=cf['bsize'], shuffle=True,pin_memory=True)
train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=cf['bsize'], shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cf['bsize'], shuffle=False)

# Define loss function
criterion = nn.BCEWithLogitsLoss()

###########################################################################
#ERM
###########################################################################
#Wandb
if cf['wb']:
    wbinit(wandb, cf, f'ERM r{run}')

model = ERMIRM(insize=cf['img_size'], hsize=cf['hsize'], outsize=cf['pred_size'])
model,optimizer,scheduler = initModel(cf['lr'],
                                      model,
                                      5*len(train_loader_1))

rERM = runERM(model, criterion, optimizer, scheduler, train_loader_1, train_loader_2, test_loader,
        cf['num_epochs'], cf['printFreq'], cf['pth'], cf['wb'], wandb)

if cf['wb']:
    wandb.finish()
###########################################################################
#IRM
###########################################################################
#Wandb
if cf['wb']:
    wbinit(wandb, cf, f'IRM r{run}')

model = ERMIRM(insize=cf['img_size'], hsize=cf['hsize'], outsize=cf['pred_size'])
model,optimizer,scheduler = initModel(cf['lr'],
                                      model,
                                      5*len(train_loader_1))

rIRM = runIRM(model, criterion, optimizer, scheduler, train_loader_1, train_loader_2, test_loader,
        cf['num_epochs'], cf['printFreq'], cf['pth'],
        cf['l2_regularizer_weight'], cf['penalty_anneal_iters'],
        cf['penlt_weight'], cf['wb'], wandb)

if cf['wb']:
    wandb.finish()
###########################################################################
#SFB
###########################################################################
cf['lr'] = 0.0003
cf['penalty_anneal_iters'] = 400
cf['l2_regularizer_weight'] = 0

#Wandb
if cf['wb']:
    wbinit(wandb, cf, f'SFB r{run}')

clas = Classifier(cf['img_size'], cf['hsize'], 8)
clas,optimizer,scheduler = initModel(cf['lr'],
                                      clas,
                                      5*len(train_loader_1))

rSFB = runSFB(clas, criterion, optimizer, scheduler, train_loader_1, train_loader_2, test_loader,
              cf['num_epochs'], cf['printFreq'], cf['pth'],
              cf['l2_regularizer_weight'], cf['penalty_anneal_iters'],
              cf['penlt_weight'], cf['base_irm_weight'],
              cf['lambdaC'], cf['gclip'], cf['lock'], cf['clip'],
              cf['sch'], cf['rescale'], cf['wb'], wandb)

if cf['wb']:
    wandb.finish()
###############################################################################

print(rERM)
print(rIRM)
print(rSFB)

'''
IRM:
Epochs: 100%|██████████| 500/500 [37:59<00:00,  4.56s/it]
Train L: 0.5989 0.5971, Test L: 0.6755, Train acc: 0.6894 0.6884, Test acc: 0.6442
ERM 0.598003 IRM 0.000009 91257.18613115903
'''