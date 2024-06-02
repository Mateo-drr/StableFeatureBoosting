# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:45:17 2024

@author: Mateo-drr
"""

from tqdm import tqdm
from utils import mean_accuracy,prepDat,fwdPass,wblog
import torch

def runERM(model,criterion,optimizer,scheduler,train_loader_1,train_loader_2,test_loader,
           num_epochs,printFreq,pth,wb,wandb,save=False):
    print('\nERM')
    # Train the model
    bestAcc=0
    bestGeo=0
    accs=[[],[]]
    
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        
        model.train()
        
        #TRAIN SET 1
        train_loss = 0.0
        for data in train_loader_1:
            img,lbl = prepDat(data)
            train_loss,ce,outputs = fwdPass(model, criterion, img, lbl, train_loss)
            acc1 = mean_accuracy(outputs, lbl)
            
        #TRAIN SET 2
        train_loss2 = 0.0
        for data in train_loader_2:
            img,lbl = prepDat(data)
            
            train_loss2,ce2,outputs = fwdPass(model, criterion, img, lbl, train_loss2)
            acc2 = mean_accuracy(outputs, lbl)
            
        # Backward pass and optimization
        loss=(ce+ce2)/2 #join both losses
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
    
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
            print(f'\nTrain Loss: {train_loss:.4f} {train_loss2:.4f}, Test Loss: {test_loss:.4f},',
                  f'Train acc: {acc1:.4f} {acc2:.4f}, Test acc: {acct:.4f}')
    
            if wb:
                wblog(wandb, ce, ce2, cet, acc1, acc2, acct, 0, 0, 0)
    
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