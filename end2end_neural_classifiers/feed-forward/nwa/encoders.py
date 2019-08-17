# -*- coding: utf-8 -*-
"""
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
    def __init__(self, input_size, embedding_size, hidden_size, opt, pretrained_embedding=None):
        # first argument: subclass, second: an instance of that subclass. Equivalent to "super()" 
        super(EncoderRNN, self).__init__()     
        self.n_layers = opt.n_layers
        self.dropout = opt.dropout
        if pretrained_embedding:
            embedding_tensor = torch.load(pretrained_embedding)
            self.embedding = nn.Embedding.from_pretrained(embedding_tensor)  # load pretrained embedding tensor 
            # fasttext embedding: dimension 100  (to override the size given by option)
            embedding_size = embedding_tensor.size(1) 
        else:
            # if there's no pretrained embeddings, the embedding_size will be 10 (given by option)
            self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=Constants.PAD)
        # first argument of nn.GRU: the number of expected features in the input x
        self.gru = nn.GRU(embedding_size, hidden_size, opt.n_layers, dropout=opt.dropout, bidirectional=True)
         
    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        # input_lengths: list of sequences lengths of each batch element.
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        # unpack (back to padded) outputs(src_len,batch,2*hidden_size) 
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs) 
        return outputs, hidden 
