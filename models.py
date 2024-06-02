# -*- coding: utf-8 -*-
"""
Created on Sun May 19 12:00:10 2024

@author: Mateo-drr
"""
import torch.nn as nn
import torch.nn.init as init

class ERMIRM(nn.Module):
    def __init__(self, insize, hsize, outsize,dout=0.2):
        super(ERMIRM, self).__init__()
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
    
#SFB
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
        self.l3 = nn.Linear(hsize, 1)
        self.l4 = nn.Linear(hsize, 1)
        self.relu = nn.LeakyReLU(inplace=True)
        
    # def initialize_weights(self):
        for lin in [self.fc1, self.fc2, self.fc3,self.fc4,self.l1,self.l2,self.l3,self.l4]:
            init.xavier_uniform_(lin.weight)
            init.zeros_(lin.bias)

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
        # pu = self.l2(pu)
        pu = self.l3(pu)
        
        p = self.relu(self.l1(x))
        #p = self.l2(p)
        p  = self.l4(p)
        return xs,xu,ps,pu,p