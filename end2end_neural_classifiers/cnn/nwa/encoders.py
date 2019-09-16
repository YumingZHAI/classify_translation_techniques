# -*- coding: utf-8 -*-
"""
Encoder class for an Automatic Classification of Translation Relations in Parallel Corpus
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


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size,opt,pretrained_embedding=None):
        super(EncoderRNN, self).__init__()        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = opt.n_layers
        self.dropout = opt.dropout
        if pretrained_embedding:
            embedding_tensor = torch.load(pretrained_embedding)
            self.embedding = nn.Embedding.from_pretrained(embedding_tensor)
            embedding_size = embedding_tensor.size(1)
        else:
            self.embedding = nn.Embedding(input_size, embedding_size,padding_idx=Constants.PAD)
        self.gru = nn.GRU(embedding_size, opt.hidden_size, opt.n_layers, dropout=opt.dropout, bidirectional=True)
         
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded) outputs(src_len,batch,2*hidden_size) 
        return outputs, hidden 
