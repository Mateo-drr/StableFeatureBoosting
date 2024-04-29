# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:45:17 2024

@author: Mateo-drr
"""

from tqdm import tqdm
from utils import mean_accuracy,prepDat,fwdPass
import torch

def runERM(model,criterion,optimizer,scheduler,train_loader_1,train_loader_2,test_loader,num_epochs,printFreq,pth,save=False):
    print('\nERM')
    # Train the model
    
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        
        model.train()
        
        #TRAIN SET 1
        train_loss = 0.0
        for data in train_loader_1:
            img,lbl = prepDat(data)
            train_loss,loss,outputs = fwdPass(model, criterion, img, lbl, train_loss)
            acc1 = mean_accuracy(outputs, lbl)
            
        #TRAIN SET 2
        train_loss2 = 0.0
        for data in train_loader_2:
            img,lbl = prepDat(data)
            
            train_loss2,loss2,outputs = fwdPass(model, criterion, img, lbl, train_loss2)
            acc2 = mean_accuracy(outputs, lbl)
            
        # Backward pass and optimization
        loss=(loss+loss2)/2 #join both losses
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
        model.eval()
        
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                img,lbl=prepDat(data)
                # Forward pass
                test_loss,_,outputs = fwdPass(model, criterion, img, lbl, test_loss)
                acct = mean_accuracy(outputs, lbl)
    
        # Print train and test progress
        if epoch % printFreq ==0 or epoch == num_epochs-1:
            train_loss /= len(train_loader_1.dataset)  # Divide by total number of samples
            train_loss2 /= len(train_loader_2.dataset)  # Divide by total number of samples
            test_loss /= len(test_loader.dataset)  # Divide by total number of samples
            print(f'\nTrain Loss: {train_loss:.4f} {train_loss2:.4f}, Test Loss: {test_loss:.4f},',
                  f'Train acc: {acc1.mean():.4f} {acc2.mean():.4f}, Test acc: {acct.mean():.4f}')
    
        if save:
            # Save the trained model
            torch.save(model.state_dict(), pth+'ERM.pth')