#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 23:21:28 2018

@author: kanwei
"""
# v2
class LSTM_EEG(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, hidden_dim2, n_class):
        super(LSTM_EEG, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm_1 = nn.LSTM(in_dim, hidden_dim, n_layer,  batch_first=True)
        self.lstm_2 = nn.LSTM(in_dim, hidden_dim, n_layer,  batch_first=True)
        self.lstm_3 = nn.LSTM(in_dim, hidden_dim, n_layer,  batch_first=True)
        self.lstm_4 = nn.LSTM(in_dim, hidden_dim, n_layer,  batch_first=True)
        self.lstm_5 = nn.LSTM(in_dim, hidden_dim, n_layer,  batch_first=True)
        
        self.outputlayer = nn.Linear(hidden_dim*5, hidden_dim2)
    
        self.classifier = nn.Linear(hidden_dim2,n_class)
        
    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
                                #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
                                #   self.hidden_dim)).cuda()
        x = x.view(-1,5,384,32)   # shape:  (batch,band,timestep,in_dim)                    
        x1 = x[:,0,:,:]# first band
        x2 = x[:,1,:,:]
        x3 = x[:,2,:,:]
        x4 = x[:,3,:,:]
        x5 = x[:,4,:,:]
        out1, _ = self.lstm_1(x1)
        out1 = out1[:, -1, :]
       
        out2, _ = self.lstm_2(x2)
        out2 = out2[:, -1, :]
        out3, _ = self.lstm_3(x3)
        out3 = out3[:, -1, :]
        out4, _ = self.lstm_4(x4)
        out4 = out4[:, -1, :]
        out5, _ = self.lstm_5(x5)
        out5 = out5[:, -1, :]
        
        out = torch.stack((out1,out2,out3,out4,out5),1)
        out = out.view(-1,5*self.hidden_dim)
     #   out = [out1,out2,out3,out4,out5]
   #     out = Variable(out)
        out =  self.outputlayer(out)
        out = F.relu(out)       
        out = self.classifier(out)
        out = F.softmax(out,dim=1)
    #   out = F.softmax(out, dim=1)
     #   return out1,out2,out3,out4,out5
        return out
