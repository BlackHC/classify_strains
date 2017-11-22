#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:35:24 2017

modified from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
"""
import torch
import torch.nn as nn
from torch.autograd import Variable

from .model_resnet import ResNetS, Bottleneck

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, num_classes, input_size, hidden_size, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        
        assert rnn_type in ['LSTM', 'GRU']
        self.rnn = getattr(nn, rnn_type)(input_size, 
                                          hidden_size, 
                                          nlayers, 
                                          dropout=dropout,
                                          batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.num_classes = num_classes

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data #this is only to get the data type of the hidden layers
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_())
    
    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        
        print(x.size(), hidden.size())
        x, _ = self.rnn(x, hidden)  
        x = self.drop(x)
        
        # Decode hidden state of last time step
        out  = self.fc(x[:, -1, :])
        return out
        
class RNNResnet(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, 
                 rnn_type, 
                 num_classes, 
                 conv_window, 
                 hidden_size, 
                 nlayers, 
                 dropout=0.5, 
                 tie_weights=False):
        super(RNNResnet, self).__init__()
        
        self.resnet = ResNetS(Bottleneck, [3, 4, 6, 3], n_channels=1, num_classes=hidden_size)
        
        self.drop = nn.Dropout(dropout)
        assert rnn_type in ['LSTM', 'GRU']
        self.rnn = getattr(nn, rnn_type)(hidden_size, 
                                          hidden_size, 
                                          nlayers, 
                                          dropout = dropout,
                                          batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        
        self.rnn_type = rnn_type
        self.conv_window = conv_window
        self.hidden_size = hidden_size
        self.nlayers = nlayers
        self.num_classes = num_classes

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data #this is only to get the data type of the hidden layers
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.nlayers, batch_size, self.hidden_size).zero_())
    
    def forward(self, X, hidden=None):
        X_c = []
        for ii in range(0, X.size(2), self.conv_window):
            x = X[:, :, ii:ii+ self.conv_window]
            x_c = self.resnet(x)
            X_c.append(torch.unsqueeze(x_c, 1))
        X_c = torch.cat(X_c, dim=1)
        
        
        if hidden is None:
            hidden = self.init_hidden(X_c.size(0))
        X_R, hidden_r = self.rnn(X_c, hidden)  
        X_R = self.drop(X_R)
        
        # Decode hidden state of last time step
        out  = self.fc(X_R[:, -1, :])
        return out, hidden_r

if __name__ == '__main__':
    mod = RNNModel(rnn_type = 'GRU',
                   num_classes = 100, 
                   input_size = 48, 
                   hidden_size = 128, 
                   nlayers = 3
                   )
    
    mod_r = RNNResnet(rnn_type = 'GRU',
                   num_classes = 100, 
                   conv_window = 200, 
                   hidden_size = 128, 
                   nlayers = 3
                   )
    #%%
    X = Variable(torch.randn(8, 1, 9000, 48).type(torch.FloatTensor))
    Y = mod_r.forward(X)
    #%%
    target = Variable(torch.range(1,8).type(torch.LongTensor))
    criterion = nn.CrossEntropyLoss()
    loss = criterion(Y, target)
    
    
    