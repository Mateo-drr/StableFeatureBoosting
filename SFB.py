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
from utils import mean_accuracy,prepDat,fwdPass,penalty, mean_nll, wblog
from models import Classifier

import torch.nn.init as init
from IRM import runIRM
from ERM import runERM
from ACTIR import runACTIR, initAdapInv, regressionLoss

import copy

def batchSFB(data,clas,ce=[],acc=[],test=False):
    img,lbl = prepDat(data)
    xs,xu,ps,pu,p = clas(img)
    # combination function see eq 4.5
    c = ps+pu-p 
    #stable risk
    sR = mean_nll(ps, lbl)
    #combined risk
    cR = mean_nll(c,lbl)
    #P stability
    if not test:
        irm = penalty(ps, lbl) # computes the CE and then adds the penalty
    else:
        irm = 0
    #P conditional independence
    actis = regressionLoss(f_beta = xs, f_eta = xu, y = lbl)
    
    #metrics
    ce.append(mean_nll(c, lbl).detach().cpu().numpy())
    acc.append(mean_accuracy(c, lbl).detach().cpu().numpy())
    if test:
        ce = ce[0]
        acc = acc[0]
    
    return sR,cR,actis,irm,ce,acc

def runSFB(clas,criterion,optimizer,scheduler,train_loader_1,train_loader_2,test_loader,
           num_epochs,printFreq,pth,l2_regularizer_weight,penalty_anneal_iters,penlt_weight,
           base_irm_weight,lambdaC,gclip,lock,clip,sch,rescale,wb,wandb):
    
    print('\nSFB')
    bestAcc=0
    bestModel=None
    
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        clas.train()    
        ce,acc,ce2,acc2 = [],[],[],[]
        sRl,cRl,pCIl,pSl = [],[],[],[]
        
        #initialize the loader for the second trainset to use nested
        tldr2 = iter(train_loader_2)
        
        for data in train_loader_1:
            
            #TRAIN SET 1
            sR,cR,actis,irm,ce,acc = batchSFB(data, clas,ce,acc)
            
            #TRAIN SET 2
            sR2,cR2,actis2,irm2,ce2,acc2 = batchSFB(next(tldr2),clas,ce2,acc2)
            
            #SFB
            #Stable risk
            sRs = (sR + sR2)/2
            #Combined risk
            cRisk = (cR+cR2)/2
            #P conditional independece aka ACTIS
            pCI = (actis + actis2)/2
            #P stability aka IRM
            pStability = torch.stack([irm,irm2]).mean()
            
            #L2 regularizer on ERM
            weight_norm = torch.tensor(0.).cuda()
            for w in clas.parameters():
                weight_norm += w.norm().pow(2)
            stableRisk = sRs.clone()
            stableRisk += l2_regularizer_weight * weight_norm
            
            #Lambda s aka IRM weight
            if epoch >= penalty_anneal_iters:
                lambdaS = epoch**1.3#penlt_weight
                crlambda = 1
                if not lock:
                    optimizer.param_groups[0]['lr'] = 0.008
                    scheduler.last_epoch = epoch
                    lock=True
            else:
                lambdaS = base_irm_weight
                crlambda = 1
        
            #COMPLETE LOSS
            sfb = stableRisk + crlambda * cRisk + lambdaS * pStability + lambdaC * pCI
            if lambdaS > 1.0 and rescale:
                # Rescale the entire loss to keep gradients in a reasonable range
                sfb /= lambdaS
                
            # Backward pass and optimization
            optimizer.zero_grad()
            sfb.backward()
            if clip:
                nn.utils.clip_grad_norm_(clas.parameters(), gclip)
            
            optimizer.step()
            if sch:
                scheduler.step()
                
            #Store the values
            sRl.append(stableRisk.detach().cpu().numpy())
            cRl.append(cRisk.detach().cpu().numpy())
            pSl.append(pStability.detach().cpu().numpy())
            pCIl.append(pCI.detach().cpu().numpy())
        
        #calculate mean values among batches
        stableRisk = np.mean(sRl)
        cRisk = np.mean(cRl)
        pStability = np.mean(pSl)
        pCI = np.mean(pCIl)
        
        #LOG VALUES
        print('sR:', round(stableRisk.item(), 3),
              'cR:', round(cRisk.item(), 3),
              'pS:', round(pStability.item(), 3), round(lambdaS,3),
              'pCI:', round(pCI.item(), 3))
    
        #TEST SET
        clas.eval()
        with torch.no_grad():
            for data in test_loader:
                img,lbl=prepDat(data)
                xs,xu,ps,pu,p = clas(img)                
                #combine
                c = ps+pu-p
                cet = mean_nll(c, lbl)
                acct = mean_accuracy(c, lbl)
    
        ce = np.mean(ce)
        ce2 = np.mean(ce2)
        acc = np.mean(acc)
        acc2 = np.mean(acc2)
        acct = acct.mean()
    
        # Print train and test progress
        if epoch % printFreq ==0 or epoch == num_epochs-1:
            print(f'\nTrain ce: {ce:.4f} {ce2:.4f}, Test ce: {cet:.4f},',
                  f'Train acc: {acc:.4f} {acc2:.4f}, Test acc: {acct:.4f}',
                  f'Train loss: {sfb:.4f}')
            if wb:
                wblog(wandb, ce, ce2, cet, acc, acc2, acct, sfb, pCI, pStability)
        
        #store best accuracy and model
        if (acc*acc2*acct)**(1/3) > bestAcc:
            bestAcc = (acc*acc2*acct)**(1/3)
            print(acc, acc2, acct.cpu().item(), epoch)
            bestModel=copy.deepcopy(clas)
    
    #Print best accuracies
    print('\n',acc, acc2, acct.cpu().item())
    return bestModel,[acc,acc2,acct.cpu().item()]

# #PARAMS
# num_epochs = 1000
# #lr = 0.005
# lr = 0.0003
# hsize=390
# path = 'D:/MachineLearning/datasets'
# pth = 'D:/Universidades/Trento/3S/AML/'
# device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bsize=25000
# img_size=2*14*14
# pred_size=1
# printFreq = 1
# #irm
# l2_regularizer_weight = 0.00110794568
# penlt_weight = 5000#91257.18613115903
# penalty_anneal_iters = 400 
# #actis
# num_env=2
# #sfb
# lambdaC = 0
# base_irm_weight = 0.1
# sch = True

# #other
# wb=True
# seed=20
# lock=False # True to not change lr
# rescale=True
# clip=False
# gclip=5
# best=0

# torch.manual_seed(seed)

# #torch.set_num_threads(8)
# #torch.set_num_interop_threads(8)
# torch.backends.cudnn.benchmark=True

# #Wandb
# if wb:
#     wandb.init(
#         # set the wandb project where this run will be logged
#         name=f'fe {seed}',
#         project="StableFeatureBoosting Clean",
    
#         # track hyperparameters and run metadata
#         config={
#         "epochs": num_epochs,
#         "learning_rate": lr,
#         'seed':seed,
#         'penlt_weight':penlt_weight,
#         'penalty_anneal_iters':penalty_anneal_iters,
#         'base_irm_weight':base_irm_weight,
#         'lambdaC':lambdaC,
#         'l2_regularizer_weight': l2_regularizer_weight,
#         }
#     )

# #def main():
# if True:
#     # Define transformations
#     transform = transforms.Compose([
#         transforms.ToTensor(), # Convert PIL image to tensor
#         #transforms.Normalize(mean=[0.5, 0.5,0.5], std=[0.5, 0.5,0.5])  # Normalization
#     ])  
    
#     # Create datasets for training and test environments
#     train_dataset_1 = dl.ColoredMNIST(root=path, env='train1', transform=transform)
#     train_dataset_2 = dl.ColoredMNIST(root=path, env='train2', transform=transform)
#     test_dataset = dl.ColoredMNIST(root=path, env='test', transform=transform)
    
#     # Create data loaders
#     train_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=bsize, shuffle=True,pin_memory=True)
#     train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=bsize, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bsize, shuffle=False)
    
#     # Define loss function
#     criterion = nn.BCEWithLogitsLoss()
    
    
#     ###########################################################################
#     #SFB
#     ###########################################################################
#     # # Define the model
#     clas = Classifier(img_size, hsize, 8)
#     clas.to(device)
    
#     optimizer = optim.AdamW(clas.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5*len(train_loader_1),eta_min=0.00001,verbose=True)
    
#     acct=torch.tensor([0.0])
    
#     def batchSFB(data,clas,ce=[],acc=[],test=False):
#         img,lbl = prepDat(data)
#         xs,xu,ps,pu,p = clas(img)
#         # combination function see eq 4.5
#         c = ps+pu-p 
#         #stable risk
#         sR = mean_nll(ps, lbl)
#         #combined risk
#         cR = mean_nll(c,lbl)
#         #P stability
#         if not test:
#             irm = penalty(ps, lbl) # computes the CE and then adds the penalty
#         else:
#             irm = 0
#         #P conditional independence
#         actis = regressionLoss(f_beta = xs, f_eta = xu, y = lbl)
        
#         #metrics
#         ce.append(mean_nll(c, lbl).detach().cpu().numpy())
#         acc.append(mean_accuracy(c, lbl).detach().cpu().numpy())
#         if test:
#             ce = ce[0]
#             acc = acc[0]
        
#         return sR,cR,actis,irm,ce,acc
    
#     for epoch in tqdm(range(num_epochs), desc='Epochs'):
#         clas.train()    
#         ce,acc,ce2,acc2 = [],[],[],[]
#         sRl,cRl,pCIl,pSl = [],[],[],[]
        
#         #initialize the loader for the second trainset to use nested
#         tldr2 = iter(train_loader_2)
        
#         for data in train_loader_1:
            
#             #TRAIN SET 1
#             sR,cR,actis,irm,ce,acc = batchSFB(data, clas,ce,acc)
            
#             #TRAIN SET 2
#             sR2,cR2,actis2,irm2,ce2,acc2 = batchSFB(next(tldr2),clas,ce2,acc2)
            
#             #SFB
#             #Stable risk
#             sRs = (sR + sR2)/2
#             #Combined risk
#             cRisk = (cR+cR2)/2
#             #P conditional independece aka ACTIS
#             pCI = (actis + actis2)/2
#             #P stability aka IRM
#             pStability = torch.stack([irm,irm2]).mean()
            
#             #L2 regularizer on ERM
#             weight_norm = torch.tensor(0.).cuda()
#             for w in clas.parameters():
#                 weight_norm += w.norm().pow(2)
#             stableRisk = sRs.clone()
#             stableRisk += l2_regularizer_weight * weight_norm
            
#             #Lambda s aka IRM weight
#             if epoch >= penalty_anneal_iters:
#                 lambdaS = epoch**1.3#penlt_weight
#                 crlambda = 1
#                 if not lock:
#                     optimizer.param_groups[0]['lr'] = 0.008
#                     scheduler.last_epoch = epoch
#                     lock=True
#             else:
#                 lambdaS = base_irm_weight
#                 crlambda = 1
        
#             #COMPLETE LOSS
#             sfb = stableRisk + crlambda * cRisk + lambdaS * pStability + lambdaC * pCI
#             if lambdaS > 1.0 and rescale:
#                 # Rescale the entire loss to keep gradients in a reasonable range
#                 sfb /= lambdaS
#             #CUSTOM UPGRADE
#             # if epoch >= penalty_anneal_iters:
#             #     pStability = torch.exp(pStability)
#             #     sfb = (stableRisk * crlambda * cRisk * pStability * lambdaC * pCI)**(1/3)
#             # else:
#             #     sfb = stableRisk + crlambda * cRisk + lambdaS*pStability + lambdaC * pCI
#             # #sfb /= torch.max(torch.tensor([stableRisk, cRisk, pStability]))
            
                
#             # Backward pass and optimization
#             optimizer.zero_grad()
#             sfb.backward()
#             if clip:
#                 nn.utils.clip_grad_norm_(clas.parameters(), gclip)
            
#             optimizer.step()
#             if sch:
#                 scheduler.step()
                
#             #Store the values
#             sRl.append(stableRisk.detach().cpu().numpy())
#             cRl.append(cRisk.detach().cpu().numpy())
#             pSl.append(pStability.detach().cpu().numpy())
#             pCIl.append(pCI.detach().cpu().numpy())
        
#         #calculate mean values among batches
#         stableRisk = np.mean(sRl)
#         cRisk = np.mean(cRl)
#         pStability = np.mean(pSl)
#         pCI = np.mean(pCIl)
#         ce = np.mean(ce)
#         ce2 = np.mean(ce2)
#         acc = np.mean(acc)
#         acc2 = np.mean(acc2)
        
#         #LOG VALUES
#         print('sR:', round(stableRisk.item(), 3),
#               'cR:', round(cRisk.item(), 3),
#               'pS:', round(pStability.item(), 3), round(lambdaS,3),
#               'pCI:', round(pCI.item(), 3))
    
#         #TEST SET
#         clas.eval()
#         with torch.no_grad():
#             for data in test_loader:
#                 #sR,cR,actis,irm,cet,acct = batchSFB(data,clas,test=True)
#                 img,lbl=prepDat(data)
                
#                 xs,xu,ps,pu,p = clas(img)
                
#                 #combine
#                 c = ps+pu-p
                
#                 cet = mean_nll(ps+pu-p, lbl)
#                 acct = mean_accuracy(ps+pu-p, lbl)
    
#         # Print train and test progress
#         if epoch % printFreq ==0 or epoch == num_epochs-1:
#             print(f'\nTrain ce: {ce:.4f} {ce2:.4f}, Test ce: {cet:.4f},',
#                   f'Train acc: {acc.mean():.4f} {acc2.mean():.4f}, Test acc: {acct.mean():.4f}',
#                   f'Train loss: {sfb:.4f}')
#             if wb:
#                 wandb.log({"Train ce1": ce,
#                        "Train ce2": ce2,
#                        "Test ce": cet,
#                        "Train acc1": acc.mean(),
#                        "Train acc2": acc2.mean(),
#                        "Test acc": acct.mean(),
#                        "SFB loss": sfb,
#                        "pCI": pCI,
#                        "pS":pStability})
        
#         #store best accuracy and model
#         acc,acc2,acct = acc.mean(),acc2.mean(),acct.mean()
#         if acct > best and (acc >= acct or acc2 >= acct):
#             print(f'Best {acc}, {acc2}, {acct}')
#             best = acct
#             bestModel = copy.deepcopy(clas)
    
    
    
#     # ###########################################################################
#     # #ACTIS
#     # ###########################################################################
#     # config = {'gamma': 0.5,
#     #           'reg_lambda':2.1544346900318834,
#     #           'reg_lambda_2':46.41588833612777,
#     #           'phi_odim':8,
#     #           }
#     # model = initAdapInv(num_env, img_size, pred_size, config['phi_odim'])
#     # model.freeze_all_but_beta()
#     # model,optimizer,scheduler = initModel(lr=0.001,model=model)
    
#     # runACTIR(model, criterion, optimizer, scheduler, train_loader_1, train_loader_2, test_loader,
#     #           num_epochs, printFreq, pth, config)

#     # ###########################################################################
#     # #ERM
#     # ###########################################################################
#     # model,optimizer,scheduler = initModel(lr)
    
#     # runERM(model, criterion, optimizer, scheduler, train_loader_1, train_loader_2, test_loader,
#     #        num_epochs, printFreq, pth)

#     # ###########################################################################
#     # #IRM
#     # ###########################################################################
#     # model,optimizer,scheduler = initModel(lr)

#     # IRMmodel = runIRM(model, criterion, optimizer, scheduler, train_loader_1, train_loader_2, test_loader,
#     #                   num_epochs, printFreq, pth, l2_regularizer_weight, penalty_anneal_iters, penlt_weight)


# ###############################################################################
# '''
# IRM:
# Epochs: 100%|██████████| 500/500 [37:59<00:00,  4.56s/it]
# Train L: 0.5989 0.5971, Test L: 0.6755, Train acc: 0.6894 0.6884, Test acc: 0.6442
# ERM 0.598003 IRM 0.000009 91257.18613115903
# '''
    
# print('''
#     Best seeds in reverse order ie w activation and then wo
#     12
#         110000 X
#         lr 0.0001 X
#         lr 0.001 X
#         lr 0.0006 X
#         lr 0.01 X
#         Adam X
#         anneal 300 X
#         anneal 64 X
#         lr 0.1 X
#         irm base 0.5 X
#         irm base 0 X
#         CI 0 X
#         CI 2 X
#         CI 1.5 X 
#         CI 1.1 X
#         CI 1.25 X
#         CI 1.25 then to 0.5 w anneal X
#         no l2reg X
#     10
#         no l2reg X
#     20
#         CI 1.25 X
#         CI 2 X
#         lr 0.001  X
#         CI 3 X
#         irm base 0.5 X
#         120000 X
#         120000 + lr 0.0006 X
#         150000 X
#         200000 X
#         1000000 X
#         lr 0.1 X
#         no l2reg :)
#         l2reg 0.0001
#         no l2reg CI 0
#         no l2reg anneal 300 
#         no l2reg 200000
#         no l2reg anneal 70 
#         no l2reg CI 3
#         no l2reg base irm 1
#         no l2reg lr 0.0006
#         no l2reg lr 0.01
#         no l2reg CI 0 indp CI 
#         no l2reg CI 0 irm e**1.6
#         no l2reg CI 0 irm e**1.2
#         no l2reg CI 0 irm e**1.9
#         no l2reg CI 0 irm e**3
#         no l2reg CI 0 irm e**3 w ann
#         no l2reg CI 0 irm e**2 w ann
#         no l2reg CI 0 irm e**2.5 w ann
#         no l2reg CI 0 irm e**2.75 w ann
#         no l2reg base irm 11697083
#         no l2reg CI 0 irm e**3.5 w ann
#         10e-5 l2reg CI 0 irm e**3 w ann 
#         no l2reg CI 0 indp CI sch
#         no l2reg CI 1 indp CI sch
#         no l2reg CI 0 irm e**3 w ann sch
#         no l2reg CI 0 irm e**3 w ann sch e1000
#         no l2reg sch e1000
#         no l2reg e1000
#         no l2reg CI 1 indp CI sch e1000
#         no l2reg CI 0->1 irm e**3 w ann sch e1000 lr 0.0006
#         no l2reg CI 0 irm e**3 w ann sch e1000 lr 0.0006
#         CI 0 sch
#         no l2reg CI 0 e1000 sch
#         no l2reg CI 0 e1000 min(e**3,irm) sch Xerror
#         no l2reg CI 1 e1000 min(e**3,irm) Xerror
#         no l2reg CI 1 e1000 min(e**2,irm) 
#         no l2reg CI 0 e1000 min(e**2,irm) lr 0.0006
#         no l2reg CI 0 e1000 sch lr 0.0006
#         no l2reg CI 0 e1000 sch lr 0.0001 -> 0.0006 Xerror
#         no l2reg CI 0 e1000 sch lr 0.0003
#         no l2reg CI 0 e1000 sch lr 0.0003 -> 0.0006
#         no l2reg CI 0 e1000 sch lr 0.0003 no div
#         no l2reg CI 0 e1000 sch lr 0.0003 div*0.8
#         no l2reg CI 0 e1000 sch lr 0.0002 -> 0.0007
#         no l2reg CI 0 e1000 sch lr 0.0003 anneal 250
#         no l2reg CI 0 e1000 min(e**2,irm) sch lr 0.0003 ? IDK ?
#         no l2reg CI 0 e1000 sch lr 0.0003 -> 0.0008
#         no l2reg CI 0 e1000 sch lr 0.0003 div with min(e**2,irm)
#         no l2reg CI 0 e1000 sch lr 0.0003 -> 0.001
#         no l2reg CI 0 e1000 sch lr 0.0003 -> 0.008 Gold!
#     18
#         no l2reg X
#     15
#         no l2reg CI 3 X
#         no l2reg indp CI X
#         no l2reg CI 0 
#         no l2reg CI 0.1 
#         no l2reg CI 0 indp CI 
#         no l2reg CI 0 indp CI *2
#         no l2reg CI 0 indp CI *5
#         no l2reg CI 0 indp CI *5 lr 0.00005
#         no l2reg CI 0 indp CI *20
#         no l2reg CI 0 indp CI lr 0.00005
#         no l2reg CI 1 indp CI 
#         no l2reg 50000 CI 0 indp CI 
#         no l2reg CI 0 indp CI irm base 1
#     13
#         no l2reg X
#         no l2reg lr 0.01 X
#         no l2reg CI 0 X
#         no l2reg lr 0.001 X
#         no l2reg CI 3 X
#     10
        
#     4
# '''    )
# print('''
#       New tests
#           random seed X
#           CI 1
#           300 anneal
#           gclip 1
#           gclip 5
#           0.0003->0.01
#           0.0003->0.05
#           seed 0 X
#           seed 1 X
#           seed 2 +- works
#       ''')
# print('''
#       seed 2 
#       seed 20 base irm 0
#       ann 400 epochs
#       ann 400 cRisk annealed
#       ann 400 cRisk annealed lr 1e-4
#       ann 400 cRisk annealed lr 1e-4 no rescale
#       ann 400 cRisk annealed lr 1e-4 no rescale base irm 0.1
      
#       Fixed!
      
#       ann 400 cRisk annealed lr 1e-4 base irm 0
#       ann 190 lr 0.0003 base irm 0.1 
#       ann 190 lr 0.0003 base irm 0.1 50000
#       ann 190 lr 0.0003 base irm 0.1 10000
#       ann 190 lr 0.0003 base irm 0.1 1
#       ann 190 lr 0.0003 base irm 0.1 100
#       ann 190 lr 0.0003 base irm 0.1 1000
#       ann 190 lr 0.0003 base irm 0.1 500
#       ann 190 lr 0.0003 base irm 0.1 5000
#       seed 0 ann 190 lr 0.0003 base irm 0.1 5000
#       seed 0 ann 190 lr 0.0003 base irm 0.1 5000 initialize
#       ann 190 lr 0.0003 base irm 0.1 5000 initialize
#       ann 190 lr 0.0003 -> 0.008 base irm 0.1 5000 initialize
#       ann 400 lr 0.0003 -> 0.008 base irm 0.1 5000 initialize
#       sed 0 ann 190 lr 0.0003 -> 0.008 base irm 0.1 5000 initialize
#       sed 0 ann 400 lr 0.0003 -> 0.008 base irm 0.1 5000 initialize 
#       ann 500 lr 0.0003 -> 0.008 base irm 0.1 5000 initialize ci 1
#       ann 500 lr 0.0003 -> 0.008 base irm 0.1 5000 initialize 
#       ann 400 lr 0.0003 -> 0.008 base irm 0.1 5000 initialize ci 1
#       ann 400 lr 0.0003 -> 0.008 base irm 0.1 5000 initialize ci 0.5
#       ann 400 lr 0.0003 -> 0.008 base irm 0.1 initialize
#       ann 400 lr 0.0003 -> 0.008 base irm 0.1 1M initialize
#       ann 400 lr 0.0003 -> 0.008 base irm 0.1 cstm initialize
#       ann 400 lr 0.0003 -> 0.008 base irm 0 e**1.6 initialize
#       ann 400 lr 0.0003 -> 0.008 base irm 0 e**3 initialize
#       ann 400 lr 0.0003 -> 0.008 base irm 0.1 cstm2 initialize
#       ann 400 lr 0.0003 -> 0.008 base irm 0.1 e**1.9 initialize
#       ann 400 lr 0.0003 -> 0.01 base irm 0.1 initialize
#       ann 400 lr 0.0003 -> 0.01 base irm 0.1 e**1.9 initialize
#       ann 400 lr 0.0003 -> 0.001 base irm 0.1 e**1.9 initialize
#       ann 400 lr 0.0003 -> 0.008 base irm 0.1 e**1.9 initAct
#       modmod ann 400 lr 0.0003 -> 0.008 base irm 0.1 e**1.9 
#       modmod ann 400 lr 0.0003 -> 0.008 base irm 0.1 e**1.9 initAct
#       modmod ann 400 lr 0.0003 -> 0.008 base irm 0.1 5000 initAct
#       modmod ann 400 lr 0.0003 -> 0.008 base irm 0.1 e**1.6 initAct
#       modmod ann 400 lr 0.0003 -> 0.008 base irm 0.1 e**1.2 initAct
#       modmod ann 400 lr 0.0003 -> 0.008 base irm 0.1 e**1.3 initAct
#       ''')


# # if __name__ == "__main__":
# #     # Call the main function
#     main()