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
from utils import mean_accuracy,prepDat,fwdPass,penalty, mean_nll

import torch.nn.init as init
from IRM import runIRM
from ERM import runERM
from ACTIR import runACTIR, initAdapInv, regressionLoss
import wandb

#PARAMS
num_epochs = 1000
#lr = 0.005
lr = 0.0003#0.0004898536566546834
hsize=390
path = 'D:/MachineLearning/datasets'
pth = 'D:/Universidades/Trento/3S/AML/'
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
bsize=25000
img_size=2*14*14
pred_size=1
printFreq = 1
#irm
l2_regularizer_weight = 0#0.00110794568
penlt_weight = 91257.18613115903
penalty_anneal_iters = 190 
#actis
num_env=2
#sfb
lambdaC = 0
base_irm_weight = 0.1
sch = True

#other
wb=True
seed=20
lock=False
rescale=True

torch.manual_seed(seed)

#torch.set_num_threads(8)
#torch.set_num_interop_threads(8)
torch.backends.cudnn.benchmark=True

#Wandb
if wb:
    wandb.init(
        # set the wandb project where this run will be logged
        name=f'def IRM {seed} !l2',
        project="StableFeatureBoosting",
    
        # track hyperparameters and run metadata
        config={
        "epochs": num_epochs,
        "learning_rate": lr,
        'seed':seed,
        'penlt_weight':penlt_weight,
        'penalty_anneal_iters':penalty_anneal_iters,
        'base_irm_weight':base_irm_weight,
        'lambdaC':lambdaC,
        'l2_regularizer_weight': l2_regularizer_weight,
        }
    )

#def main():
if True:
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert PIL image to tensor
        #transforms.Normalize(mean=[0.5, 0.5,0.5], std=[0.5, 0.5,0.5])  # Normalization
    ])  
    
    # Create datasets for training and test environments
    train_dataset_1 = dl.ColoredMNIST(root=path, env='train1', transform=transform)
    train_dataset_2 = dl.ColoredMNIST(root=path, env='train2', transform=transform)
    test_dataset = dl.ColoredMNIST(root=path, env='test', transform=transform)
    
    # Create data loaders
    train_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=bsize, shuffle=True,pin_memory=True)
    train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=bsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bsize, shuffle=False)
    
    # Define the model
    class SimpleModel(nn.Module):
        def __init__(self, insize, hsize, outsize,dout=0.2):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(insize, hsize)
            self.fc2 = nn.Linear(hsize, hsize)
            self.fc3 = nn.Linear(hsize, outsize)
            self.drop = nn.Dropout(dout)
            self.relu = nn.LeakyReLU(inplace=True)
            self.insize=insize
            
        #def initialize_weights(self):
            for lin in [self.fc1, self.fc2, self.fc3]:
                init.xavier_uniform_(lin.weight)
                init.zeros_(lin.bias)
    
        def forward(self, x):
            x = x.reshape([-1,self.insize])
            #x = self.drop(x)
            x = self.relu(self.fc1(x))
            #x = self.drop(x)
            #x = self.relu(self.fc2(x))
            #x = self.drop(x)
            x = self.fc3(x)
            return x#F.sigmoid(x)
    
    
    # Define loss function
    criterion = nn.BCEWithLogitsLoss()
    
    def initModel(lr,model=None):
        if model is None:
            # Instantiate the model
            model = SimpleModel(insize=img_size, hsize=hsize, outsize=pred_size)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        # Define the learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5*len(train_loader_1),eta_min=0.00001,verbose=True)
        return model,optimizer,scheduler
    
    
    ###########################################################################
    #SFB
    ###########################################################################
    # # Define the model
    
    class Classifier(nn.Module):
        def __init__(self, insize, hsize, outsize,dout=0.2):
            super(Classifier, self).__init__()
            
            #encoder
            self.fc1 = nn.Linear(insize, hsize)
            self.fc2 = nn.Linear(outsize, outsize)
            self.fc3 = nn.Linear(outsize, outsize)
            self.fc4 = nn.Linear(hsize, outsize)
            self.relu = nn.LeakyReLU(inplace=True)
            self.d = nn.Dropout(0.2)
            self.insize=insize
            
            #classifier
            self.l1 = nn.Linear(outsize, hsize)
            self.l2 = nn.Linear(hsize, 1)
            self.relu = nn.LeakyReLU(inplace=True)
            
        #def initialize_weights(self):
            # for lin in [self.fc1, self.fc2, self.fc3,self.fc4,self.l1,self.l2]:
            #     init.xavier_uniform_(lin.weight)
            #     init.zeros_(lin.bias)
    
        def forward(self, x):
            
            x = x.reshape([-1,self.insize])
            x = self.relu(self.fc1(x))
            x = self.d(x)
            x = self.relu(self.fc4(x))
            xs = self.fc3(x)
            xu = self.fc2(x)
            #x = self.fc4(xs + xu)
            
            ps = self.relu(self.l1(xs))
            ps = self.l2(ps)
            
            pu = self.relu(self.l1(xu))
            pu = self.l2(pu)
            
            p = self.relu(self.l1(x))
            p = self.l2(p)
            return xs,xu,ps,pu,p
    

    clas = Classifier(img_size, hsize, 8)
    clas.to(device)
    
    #PARAMS
    
    #penlt_weight=110000
    #penalty_anneal_iters=190
    #lr = 0.0001
    #num_epochs = 600
    
    optimizer = optim.AdamW(clas.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5*len(train_loader_1),eta_min=0.00001,verbose=True)
    
    acct=torch.tensor([0.0])
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        
        clas.train()
        
        #TRAIN SET 1

        for data in train_loader_1:
            img,lbl = prepDat(data)
            

            xs,xu,ps,pu,p = clas(img)
            
            #Join predictions
            c = ps+pu-p # combination function see eq 4.5
            
            #stable risk
            sR = mean_nll(ps, lbl)
            #combined risk
            c = mean_nll(c,lbl)
            
            #P stability
            irm = penalty(ps, lbl) # computes the CE and then adds the penalty
            
            #P conditional independence
            actis = regressionLoss(f_beta = xs, f_eta = xu, y = lbl)
            
            #metrics
            ce1 = mean_nll(ps+pu-p, lbl)
            acc1 = mean_accuracy(ps+pu-p, lbl)
            
        #TRAIN SET 2

        for data in train_loader_2:
            img,lbl = prepDat(data)

            xs,xu,ps,pu,p = clas(img)
            
            #Join predictions
            c2 = ps+pu-p # combination function see eq 4.5
            
            #stable risk
            sR2 = mean_nll(ps, lbl)
            #combined risk
            c2 = mean_nll(c2,lbl)
            
            #P stability
            irm2 = penalty(ps, lbl) # computes the CE and then adds the penalty

            #P conditional independence
            actis2 = regressionLoss(f_beta = xs, f_eta = xu, y = lbl)
            
            #metrics
            ce2 = mean_nll(ps+pu-p, lbl)
            acc2 = mean_accuracy(ps+pu-p, lbl)
            
        #SFB
        
        #joining from both environments
        stableRisk = (sR + sR2)/2
        cRisk = (c+c2)/2
        
        #P conditional independece aka ACTIS
        pCI = (actis + actis2)
        
        #P stability aka IRM
        train_penalty = torch.stack([irm,irm2]).mean()
        #irm - prevent overfitting
        weight_norm = torch.tensor(0.).cuda()
        for w in clas.parameters():
            weight_norm += w.norm().pow(2)
        pStability = stableRisk.clone()
        pStability += l2_regularizer_weight * weight_norm
        
        #Lambda s aka IRM weight
        #lambdaS = (penlt_weight if epoch >= penalty_anneal_iters else base_irm_weight)
        if epoch >= penalty_anneal_iters:
            lambdaS = penlt_weight
            #lambdaS = epoch ** 3
            #lambdaS = min(epoch**2, penlt_weight)
            #lambdaC = penlt_weight
            if not lock:
                optimizer.param_groups[0]['lr'] = 0.008
                scheduler.last_epoch = epoch
                lock=True
        else:
            lambdaS = base_irm_weight
        
            
        # if acct.mean() >= 0.5:
            # lambdaS = penalty_anneal_iters*2
        # if epoch >= penalty_anneal_iters:
        #     lambdaC = 0.5

        
        #COMPLETE LOSS
        sfb = stableRisk + cRisk + lambdaS * pStability + lambdaC * pCI
        
        if lambdaS > 1.0 and rescale:
            # Rescale the entire loss to keep gradients in a reasonable range
            sfb /= lambdaS
            #sfb /= min(epoch**2, penlt_weight)
            
        # Backward pass and optimization
        optimizer.zero_grad()
        sfb.backward()
        optimizer.step()
        if sch:
            scheduler.step()
        
        print('sR:', round(stableRisk.item(), 3),
              'cR:', round(cRisk.item(), 3),
              'pS:', round(pStability.item(), 3), round(lambdaS,3),
              'pCI:', round(pCI.item(), 3))

    
        clas.eval()
        
        with torch.no_grad():
            for data in test_loader:
                img,lbl=prepDat(data)
                
                xs,xu,ps,pu,p = clas(img)
                
                #combine
                c = ps+pu-p
                
                cet = mean_nll(ps+pu-p, lbl)
                acct = mean_accuracy(ps+pu-p, lbl)
    
        # Print train and test progress
        if epoch % printFreq ==0 or epoch == num_epochs-1:
            #ce1 /= len(train_loader_1.dataset)  # Divide by total number of samples
            #ce2 /= len(train_loader_2.dataset)  # Divide by total number of samples
            #cet /= len(test_loader.dataset)  # Divide by total number of samples
            print(f'\nTrain ce: {ce1:.4f} {ce2:.4f}, Test ce: {cet:.4f},',
                  f'Train acc: {acc1.mean():.4f} {acc2.mean():.4f}, Test acc: {acct.mean():.4f}',
                  f'Train loss: {sfb:.4f}')
            if wb:
                wandb.log({"Train ce1": ce1,
                       "Train ce2": ce2,
                       "Test ce": cet,
                       "Train acc1": acc1.mean(),
                       "Train acc2": acc2.mean(),
                       "Test acc": acct.mean(),
                       "SFB loss": sfb,
                       "pCI": pCI})
    
    
    
    # ###########################################################################
    # #ACTIS
    # ###########################################################################
    # config = {'gamma': 0.5,
    #           'reg_lambda':2.1544346900318834,
    #           'reg_lambda_2':46.41588833612777,
    #           'phi_odim':8,
    #           }
    # model = initAdapInv(num_env, img_size, pred_size, config['phi_odim'])
    # model.freeze_all_but_beta()
    # model,optimizer,scheduler = initModel(lr=0.001,model=model)
    
    # runACTIR(model, criterion, optimizer, scheduler, train_loader_1, train_loader_2, test_loader,
    #           num_epochs, printFreq, pth, config)

    # ###########################################################################
    # #ERM
    # ###########################################################################
    # model,optimizer,scheduler = initModel(lr)
    
    # runERM(model, criterion, optimizer, scheduler, train_loader_1, train_loader_2, test_loader,
    #        num_epochs, printFreq, pth)

    # ###########################################################################
    # #IRM
    # ###########################################################################
    # model,optimizer,scheduler = initModel(lr)

    # IRMmodel = runIRM(model, criterion, optimizer, scheduler, train_loader_1, train_loader_2, test_loader,
    #                   num_epochs, printFreq, pth, l2_regularizer_weight, penalty_anneal_iters, penlt_weight)


###############################################################################
'''
IRM:
Epochs: 100%|██████████| 500/500 [37:59<00:00,  4.56s/it]
Train L: 0.5989 0.5971, Test L: 0.6755, Train acc: 0.6894 0.6884, Test acc: 0.6442
ERM 0.598003 IRM 0.000009 91257.18613115903
'''
    
print('''
    Best seeds in reverse order ie w activation and then wo
    12
        110000 X
        lr 0.0001 X
        lr 0.001 X
        lr 0.0006 X
        lr 0.01 X
        Adam X
        anneal 300 X
        anneal 64 X
        lr 0.1 X
        irm base 0.5 X
        irm base 0 X
        CI 0 X
        CI 2 X
        CI 1.5 X 
        CI 1.1 X
        CI 1.25 X
        CI 1.25 then to 0.5 w anneal X
        no l2reg X
    10
        no l2reg X
    20
        CI 1.25 X
        CI 2 X
        lr 0.001  X
        CI 3 X
        irm base 0.5 X
        120000 X
        120000 + lr 0.0006 X
        150000 X
        200000 X
        1000000 X
        lr 0.1 X
        no l2reg :)
        l2reg 0.0001
        no l2reg CI 0
        no l2reg anneal 300 
        no l2reg 200000
        no l2reg anneal 70 
        no l2reg CI 3
        no l2reg base irm 1
        no l2reg lr 0.0006
        no l2reg lr 0.01
        no l2reg CI 0 indp CI 
        no l2reg CI 0 irm e**1.6
        no l2reg CI 0 irm e**1.2
        no l2reg CI 0 irm e**1.9
        no l2reg CI 0 irm e**3
        no l2reg CI 0 irm e**3 w ann
        no l2reg CI 0 irm e**2 w ann
        no l2reg CI 0 irm e**2.5 w ann
        no l2reg CI 0 irm e**2.75 w ann
        no l2reg base irm 11697083
        no l2reg CI 0 irm e**3.5 w ann
        10e-5 l2reg CI 0 irm e**3 w ann 
        no l2reg CI 0 indp CI sch
        no l2reg CI 1 indp CI sch
        no l2reg CI 0 irm e**3 w ann sch
        no l2reg CI 0 irm e**3 w ann sch e1000
        no l2reg sch e1000
        no l2reg e1000
        no l2reg CI 1 indp CI sch e1000
        no l2reg CI 0->1 irm e**3 w ann sch e1000 lr 0.0006
        no l2reg CI 0 irm e**3 w ann sch e1000 lr 0.0006
        CI 0 sch
        no l2reg CI 0 e1000 sch
        no l2reg CI 0 e1000 min(e**3,irm) sch Xerror
        no l2reg CI 1 e1000 min(e**3,irm) Xerror
        no l2reg CI 1 e1000 min(e**2,irm) 
        no l2reg CI 0 e1000 min(e**2,irm) lr 0.0006
        no l2reg CI 0 e1000 sch lr 0.0006
        no l2reg CI 0 e1000 sch lr 0.0001 -> 0.0006 Xerror
        no l2reg CI 0 e1000 sch lr 0.0003
        no l2reg CI 0 e1000 sch lr 0.0003 -> 0.0006
        no l2reg CI 0 e1000 sch lr 0.0003 no div
        no l2reg CI 0 e1000 sch lr 0.0003 div*0.8
        no l2reg CI 0 e1000 sch lr 0.0002 -> 0.0007
        no l2reg CI 0 e1000 sch lr 0.0003 anneal 250
        no l2reg CI 0 e1000 min(e**2,irm) sch lr 0.0003 ? IDK ?
        no l2reg CI 0 e1000 sch lr 0.0003 -> 0.0008
        no l2reg CI 0 e1000 sch lr 0.0003 div with min(e**2,irm)
        no l2reg CI 0 e1000 sch lr 0.0003 -> 0.001
        no l2reg CI 0 e1000 sch lr 0.0003 -> 0.008
    18
        no l2reg X
    15
        no l2reg CI 3 X
        no l2reg indp CI X
        no l2reg CI 0 
        no l2reg CI 0.1 
        no l2reg CI 0 indp CI 
        no l2reg CI 0 indp CI *2
        no l2reg CI 0 indp CI *5
        no l2reg CI 0 indp CI *5 lr 0.00005
        no l2reg CI 0 indp CI *20
        no l2reg CI 0 indp CI lr 0.00005
        no l2reg CI 1 indp CI 
        no l2reg 50000 CI 0 indp CI 
        no l2reg CI 0 indp CI irm base 1
    13
        no l2reg X
        no l2reg lr 0.01 X
        no l2reg CI 0 X
        no l2reg lr 0.001 X
        no l2reg CI 3 X
    10
        
    4
'''    )


# if __name__ == "__main__":
#     # Call the main function
#     main()