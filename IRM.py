# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:43:11 2024

@author: Mateo-drr
"""

import torch
from utils import prepDat, penalty, mean_accuracy, fwdPass, wblog
from tqdm import tqdm

def runIRM(model,criterion,optimizer,scheduler,train_loader_1,train_loader_2,test_loader,
           num_epochs,printFreq,pth,l2_regularizer_weight,penalty_anneal_iters,penlt_weight,
           wb,wandb,save=False):
    print('\nIRM')
    
    bestAcc=0
    bestGeo=0
    accs=[[],[]]

    # Train the model
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        
        model.train()
        
        #TRAIN SET 1
        train_loss = 0.0
        for data in train_loader_1:
            img,lbl = prepDat(data)
            train_loss,ce,outputs = fwdPass(model, criterion, img, lbl, train_loss, op='IRM')
            acc1 = mean_accuracy(outputs, lbl)
            penlt = penalty(outputs, lbl)
            
        #TRAIN SET 2
        train_loss2 = 0.0
        for data in train_loader_2:
            img,lbl = prepDat(data)
            
            train_loss2,ce2,outputs = fwdPass(model, criterion, img, lbl, train_loss2, op='IRM')
            acc2 = mean_accuracy(outputs, lbl)
            penlt2 = penalty(outputs, lbl)
            
        #IRM code
        train_nll= torch.stack([ce, ce2]).mean() #join both losses
        #train_acc = torch.stack([acc1, acc2]).mean() #join both acc
        train_penalty = torch.stack([penlt, penlt2]).mean() #join both penalties
        
        
        weight_norm = torch.tensor(0.).cuda()
        for w in model.parameters():
            weight_norm += w.norm().pow(2)
    
        loss = train_nll.clone()
        loss += l2_regularizer_weight * weight_norm
        penalty_weight = (penlt_weight if epoch >= penalty_anneal_iters else 0.1) #2.5
        #penalty_weight = (epoch+1)**2.5
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        #print('debug', train_nll.item(), train_penalty.item(), l2_regularizer_weight*weight_norm.item())
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
    
        model.eval()
        
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                img,lbl=prepDat(data)
                # Forward pass
                test_loss,cet,outputs = fwdPass(model, criterion, img, lbl, test_loss)
                acct = mean_accuracy(outputs, lbl)
    
        # Print train and test progress
        acc1,acc2,acct = acc1.mean(),acc2.mean(),acct.mean()
        if epoch % printFreq ==0 or epoch == num_epochs-1:
            train_loss /= len(train_loader_1.dataset)  # Divide by total number of samples
            train_loss2 /= len(train_loader_2.dataset)  # Divide by total number of samples
            test_loss /= len(test_loader.dataset)  # Divide by total number of samples
            print(f'\nTrain L: {train_loss:.4f} {train_loss2:.4f}, Test L: {test_loss:.4f},',
                  f'Train acc: {acc1:.4f} {acc2:.4f}, Test acc: {acct:.4f}')
            print(f'ERM {train_nll.item():.6f} IRM {loss.item():.6f}', penalty_weight)
            
            if wb:
                wblog(wandb, ce, ce2, cet, acc1, acc2, acct, 0, 0, train_penalty)
        
        if save:
            # Save the trained model
            torch.save(model.state_dict(), pth+'ERM.pth')
            
        if (acc1*acc2*acct)**(1/3) > bestGeo:
            a1,a2,at = acc1.cpu().item(),acc2.cpu().item(),acct.cpu().item()
            bestGeo = (a1*a2*at)**(1/3)
            print(a1, a2, at, epoch)
            accs[0] = [a1,a2,at]
        
        if acct > bestAcc:
            acc1,acc2,acct = acc1.cpu().item(),acc2.cpu().item(),acct.cpu().item()
            bestAcc = acct
            print(acc1, acc2, acct, epoch)
            accs[1] = [acc1,acc2,acct]

    return accs