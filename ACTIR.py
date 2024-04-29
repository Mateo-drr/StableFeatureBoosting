# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:58:47 2024

@author: Mateo-drr
"""

import torch
from torch import nn
import copy
from tqdm import tqdm
import torch.nn.functional as F
from utils import DiscreteConditionalExpecationTest, prepDat, mean_nll, mean_accuracy
from torch.autograd import grad

class AdapInv(nn.Module):
    def __init__(self, n_batch_envs, input_dim, Phi, classification=True, out_dim=1, phi_dim = None):
        super(AdapInv, self).__init__()

        self.n_batch_envs = n_batch_envs 
        self.input_dim = input_dim
        self.classification = classification
        
        # Define \Phi aka the classifier
        self.Phi = copy.deepcopy(Phi)
        self.phi_odim = phi_dim
        
        # Define \beta aka linear transformation to the output of phi
        self.beta = torch.nn.Parameter(torch.zeros(self.phi_odim, out_dim), requires_grad = False) 
        # Identity matrix initialization -> keeps the features of phi
        for i in range(out_dim):
          self.beta[i,i] = 1.0
    
        # Define \eta aka transformation to adapt phi to each environment
        self.etas = nn.ParameterList([torch.nn.Parameter(torch.zeros(self.phi_odim, out_dim), requires_grad = True) for i in range(n_batch_envs)]) 
    
        self.softmax_layer = nn.Softmax(dim=-1)
        
    def forward(self, x, env_ind, rep_learning = False, fast_eta = None):
        if rep_learning:
          rep = x
        else:
          rep = self.Phi(x)
    
        f_beta = rep @ self.beta #multiply beta with the output of phi
        if fast_eta is None:
          f_eta = rep @ self.etas[env_ind] #multiply the output of phi with the corresponding env eta
        else:
          f_eta = rep @ fast_eta[0]
    
        return f_beta, f_eta, rep #return the Bxphi, Nxphi and phi

    def sample_base_classifer(self, x):
        x_tensor = torch.Tensor(x)
        return self.Phi(x_tensor) @ self.beta

    """ used to free and check var """
    def freeze_all_but_etas(self):
      for para in self.parameters():
        para.requires_grad = False
  
      for eta in self.etas:
        eta.requires_grad = True
  
    def set_etas_to_zeros(self):
      # etas are only temporary and should be set to zero during test
      for eta in self.etas:
        eta.zero_()
  
    def freeze_all_but_phi(self):
      for para in self.parameters():
        para.requires_grad = True
  
      for eta in self.etas:
        eta.requires_grad = False
      
      self.beta.requires_grad = False
  
    def freeze_all_but_beta(self):
      for para in self.parameters():
        para.requires_grad = True
      
      self.beta.requires_grad = False
  
    def freeze_all(self):
      for para in self.parameters():
        para.requires_grad = False
  
    def free_all(self):
      for para in self.parameters():
        para.requires_grad = True
  
    def check_var_with_required_grad(self):
      """ Check what paramters are required grad """
      for name, param in self.named_parameters():
        if param.requires_grad:print(name)

class Net(nn.Module):
    def __init__(self, phi_odim):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 20, 5, 1)
        self.conv2 = nn.Conv2d(20,50,2,stride=1)#nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, phi_odim)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #[b,20,10,10]
        x = F.max_pool2d(x, 2, 2) #[b,20,5,5]
        x = F.relu(self.conv2(x)) #[b,50,1,1]
        #x = F.max_pool2d(x, 2, 2) 
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def regressionLoss(f_beta, f_eta, y, classification=True):
    
    if classification:
        reg_loss = torch.sum(torch.abs(DiscreteConditionalExpecationTest(f_beta, f_eta, y)))
    else:
        pass #some more code removed for clarity

    return reg_loss

def contraintLoss(f_beta, f_eta, y, env_ind, criterion, reg_lambda):

    #conditional independence
    reg_loss = regressionLoss(f_beta, f_eta, y)
    #section 4.3 L e inner ---> so we have =>   loss + lambda Ci
    loss = criterion(f_beta + f_eta, y) + reg_lambda * reg_loss
    
    return loss

def initAdapInv(num_env,img_size,pred_size,phi_odim):
    #Define feature extractor
    Phi = Net(phi_odim)
    #Define the model
    model = AdapInv(num_env, img_size, Phi, classification=True,
                    out_dim = pred_size, phi_dim = phi_odim)
    return model
    
def calcLoss(img,lbl,f_beta,f_eta,env_ind,reg_lambda,reg_lambda_2,gamma,criterion,
             model,phi_loss,base_loss,loss,total):
    
    #section 4.3 L e inner
    contraint_loss = contraintLoss(f_beta, f_eta, lbl, env_ind, criterion, reg_lambda)
    #Now take the gradient
    gradient = grad(contraint_loss, model.etas[env_ind], create_graph=True)[0].pow(2).mean()
    
    #section 4.3 L(wb,we,phi) = sum gamma * loss  +  (1-gamma) * loss  -> first term of eq
    phi_loss += gamma * criterion(f_beta + f_eta, lbl) + (1 - gamma) * criterion(f_beta, lbl) 
    # + lambda * grad(L e inner) -> second term of eq
    phi_loss += reg_lambda_2 * gradient
                
    #This code calculates accuracy but im using the one from IRM
    #_, base_predicted = torch.max(f_beta.data, 1)
    #base_correct_or_not = base_predicted == lbl
    #base_loss += (base_predicted == lbl).sum()
    #base_all_prediction += (base_correct_or_not).cpu().numpy().tolist()
    
    #_, predicted = torch.max((f_beta + f_eta).data, 1)
    #loss += (predicted == lbl).sum()
    total += lbl.size(0)
    return phi_loss,base_loss,loss,total

def runACTIR(model,criterion,optimizer,scheduler,train_loader_1,train_loader_2,
             test_loader,num_epochs,printFreq,pth,config,save=False):
    print('\nACTIR')
    
    # Train the model
    
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        
        model.train()
        phi_loss = 0
        total = 0
        base_loss = 0
        loss = 0
        
        #TRAIN SET 1
        env_ind=0
        for data in train_loader_1:
            img,lbl = prepDat(data)
            #beta is general, eta is domain specific
            f_beta, f_eta, _ = model(img, env_ind)
            
            phi_loss,base_loss,loss,total = calcLoss(img, lbl, f_beta, f_eta, env_ind, config['reg_lambda'], 
                                                     config['reg_lambda_2'], config['gamma'], criterion, model,
                                                     phi_loss,base_loss,loss,total)
            #Extra loss for comparison with other methods
            ceLoss = mean_nll(f_beta+f_eta, lbl)
            train_loss= ceLoss* f_beta.size(0)
            acc1 = mean_accuracy(f_beta, lbl)
            acc1e = mean_accuracy(f_beta+f_eta, lbl)
            #print('aa', mean_accuracy(f_beta, lbl, notLogit=True))
            #print('aa', mean_accuracy(f_beta+f_eta, lbl, notLogit=True))
            
        #TRAIN SET 2
        env_ind=1
        for data in train_loader_2:
            img,lbl = prepDat(data)
            f_beta, f_eta, _ = model(img, env_ind)
            
            phi_loss,base_loss,loss,total = calcLoss(img, lbl, f_beta, f_eta, env_ind, config['reg_lambda'], 
                                                     config['reg_lambda_2'], config['gamma'], criterion, model,
                                                     phi_loss,base_loss,loss,total)
            
        #TODO - test accuracy not correct
            #Extra loss for comparison with other methods
            ceLoss = mean_nll(f_beta+f_eta, lbl)
            train_loss2 = ceLoss* f_beta.size(0)
            acc2 = mean_accuracy(f_beta, lbl)
            acc2e = mean_accuracy(f_beta+f_eta, lbl)
            #print('aa', mean_accuracy(f_beta, lbl, notLogit=True))
            #print('aa', mean_accuracy(f_beta+f_eta, lbl, notLogit=True))
            
        # Backward pass and optimization
        optimizer.zero_grad()
        phi_loss.backward()
        optimizer.step()
    
        #TEST SET
        model.eval()
        test_loss = 0.0
        env_ind=0
        with torch.no_grad():
            for data in test_loader:
                img,lbl=prepDat(data)
                # Forward pass
                f_beta, f_eta, _ = model(img, env_ind)
                ceLoss = mean_nll(f_beta+f_eta, lbl)
                test_loss = ceLoss* f_beta.size(0)
                acct = mean_accuracy(f_beta, lbl)
                accte = mean_accuracy(f_beta+f_eta, lbl)
                #print('aa', mean_accuracy(f_beta, lbl, notLogit=True))
                #print('aa', mean_accuracy(f_beta+f_eta, lbl, notLogit=True))
                #actis,_,_,_ = calcLoss(img, lbl, f_beta, f_eta, env_ind, config['reg_lambda'], 
                 #                                        config['reg_lambda_2'], config['gamma'], criterion, model,
                  #                                       0,0,0,0)
    
        # Print train and test progress
        if epoch % printFreq ==0 or epoch == num_epochs-1:
            train_loss /= len(train_loader_1.dataset)  # Divide by total number of samples
            train_loss2 /= len(train_loader_2.dataset)  # Divide by total number of samples
            test_loss /= len(test_loader.dataset)  # Divide by total number of samples
            phiL = phi_loss.item()/(2*len(train_loader_2.dataset))
            
            print('\nSET 1:\n',
                  'g(X) aka Wb:',acc1.item(),'\n',
                  'f(x) aka We+Wb:', acc1e.item(),'\n',
                  'CE', train_loss.item())
            print('SET 2:\n',
                  'g(X) aka Wb:',acc2.item(),'\n',
                  'f(x) aka We+Wb:', acc2e.item(),'\n',
                  'CE', train_loss2.item(), 'ActisL', phi_loss.item())
                
            print('Test:\n',
                  'g(X) aka Wb:',acct.item(),'\n',
                  'f(x) aka We+Wb:', accte.item(),'\n',
                  'CE', test_loss.item(), '\n')#'ActisL', actis)
            
            # print(f'\nTrain Loss: {train_loss:.4f} {train_loss2:.4f}, Test Loss: {test_loss:.4f}\n',
            #       f'F acc1 {acc1e:.4f} acc2 {acc2e:.4f} Phi loss {phiL}\n' ,
            #       f'Train acc: {acc1.mean():.4f} {acc2.mean():.4f}, Test acc: {acct.mean():.4f} BetaEta {accte:.4f}\n')
    
        if save:
            # Save the trained model
            torch.save(model.state_dict(), pth+'ERM.pth')
            
    #Fine tune on 