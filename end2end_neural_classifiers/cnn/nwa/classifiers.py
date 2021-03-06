# -*- coding: utf-8 -*-
"""
classifiers class for an Automatic Classification of Translation Relations in Parallel Corpus
*****************************************************************************************
Author: Pooyan SAFARI 
"""
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import Constants
import pdb


class classifierMLP(nn.Module):
    def __init__(self, input_size,hidden_size,output_size,dropout=0.1):
        super(classifierMLP, self).__init__()        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, output_size)
         
    def forward(self, src_vectors,tgt_vectors): 
        classifier_input = torch.cat((src_vectors,tgt_vectors),1)
        hidden_layer1 = self.layer1(classifier_input)
        hidden_layer1 = self.dropout(hidden_layer1)
        outputs = self.out(hidden_layer1) 
        return outputs

class classifierCNN(nn.Module):
    def __init__(self,dropout=.1):
        super(classifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        #self.conv2 = nn.Conv2d(2, 4, 1)
        #self.conv3 = nn.Conv2d(4, 8, 1)
        #self.conv4 = nn.Conv2d(8, 16, 1)
        #self.conv5 = nn.Conv2d(16, 32, 1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(32, 100)
        self.dropout = nn.Dropout(dropout)
        #self.fc = nn.Linear(32, 4)
        self.fc = nn.Linear(100, 2)
    
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        #x = F.relu(self.conv5(x))
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        x = self.fc(x)
        return x
