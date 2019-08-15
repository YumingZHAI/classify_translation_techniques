# -*- coding: utf-8 -*-

"""
Author: Pooyan Safari
"""

from __future__ import unicode_literals, print_function, division
from io import open
import sys
import unicodedata
import string
import re
import random
import argparse
import time
import math
import pickle
import pdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

from DataLoader import DataLoader
import Constants
from encoders import *
from classifiers import *
from masked_cross_entropy import *

plt.switch_backend('agg')

###### I'm here
def trainBatch(src_batch, src_lengths, src_lengths_sorted_idx,tgt_batch, tgt_lengths,tgt_lengths_sorted_idx ,labels,mask_src, mask_tgt, encoder_src, encoder_tgt, classifier,encoder_src_optimizer, encoder_tgt_optimizer, classifier_optimizer,criterion,opt):
    ''' operation on each mini-batch in training phase '''    
    # Turn padded array tgt_batch(batch_size,tgt_len) tensors, into target_var(tgt_len,batch_size)    
    if opt.cuda:
        input_var = Variable(torch.cuda.LongTensor(src_batch)).transpose(0, 1)
        target_var = Variable(torch.cuda.LongTensor(tgt_batch)).transpose(0, 1)
        label_var = Variable(torch.cuda.LongTensor(labels))
    else:
        input_var = Variable(torch.LongTensor(src_batch)).transpose(0, 1)
        target_var = Variable(torch.LongTensor(tgt_batch)).transpose(0, 1)
        # remove the .cuda here to run on CPU
        label_var = Variable(torch.LongTensor(labels))

    # Zero gradients of both optimizers
    encoder_src_optimizer.zero_grad()
    encoder_tgt_optimizer.zero_grad()
    classifier_optimizer.zero_grad()
    loss = 0  
    # Run words through encoder
    encoder_src_outputs, encoder_src_hidden = encoder_src(input_var, src_lengths) #encoder_src_outputs(src_len+</s>,batch,hidden_size) encoder_src_hidden(num_directions,batch,hidden_size) decoder_hidden(1,batch,hidden_size)
    encoder_tgt_outputs, encoder_tgt_hidden = encoder_tgt(target_var, tgt_lengths) #encoder_tgt_outputs(src_len+</s>,batch,hidden_size) encoder_tgt_hidden(num_directions,batch,hidden_size) decoder_hidden(1,batch,hidden_size)
    #apply averaging here
    encoder_src_outputs = encoder_src_outputs.sum(0,keepdim=True) / src_lengths.unsqueeze(1).expand(1,encoder_src_outputs.size(1),encoder_src_outputs.size(2)).float() #encoder_src_outputs(1,batch,hidden_size)
    encoder_tgt_outputs = encoder_tgt_outputs.sum(0,keepdim=True) / tgt_lengths.unsqueeze(1).expand(1,encoder_tgt_outputs.size(1),encoder_tgt_outputs.size(2)).float() #encoder_tgt_outputs(1,batch,hidden_size)
    #reorder to original 
    encoder_src_outputs = encoder_src_outputs[:,src_lengths_sorted_idx.sort()[1],:].permute(1,0,2).squeeze(1) #encoder_src_outputs(batch,hidden_size)
    encoder_tgt_outputs = encoder_tgt_outputs[:,tgt_lengths_sorted_idx.sort()[1],:].permute(1,0,2).squeeze(1) #encoder_tgt_outputs(batch,hidden_size)
    
    #classifier
    pred = classifier(encoder_src_outputs.clone(),encoder_tgt_outputs.clone())
    #Loss calculation and backpropagation
    #label_var_onehot = torch.zeros(pred.size(0),pred.size(1)).cuda() #debug
    #for i in range(2): #debug
    #    if i==0:
    #        label_var_onehot[label_var==0,i]=1 #debug
    #    else:
    #        label_var_onehot[label_var==1,i]=1 #debug
    #loss = criterion(pred.clone(),label_var_onehot.clone())
    loss = criterion(pred.clone(),label_var)
    
    loss.backward()

    # clip_grad_norm for gradient clipping
    if opt.clip!=0:
        torch.nn.utils.clip_grad_norm(encoder_src.parameters(), opt.clip) #turn off for debugging
        torch.nn.utils.clip_grad_norm(encoder_tgt.parameters(), opt.clip) #turn off for debugging
    
    # Update parameters with optimizers
    encoder_src_optimizer.step()
    encoder_tgt_optimizer.step()
    classifier_optimizer.step()
    return loss

def trainEpoch(training_data,encoder_src, encoder_tgt,classifier,encoder_src_optimizer,encoder_tgt_optimizer,classifier_optimizer,criterion,opt):
    ''' Epoch operation in training phase '''
    encoder_src.train()
    encoder_tgt.train()
    classifier.train()
    total_loss = 0
    batch_num = 0
    for batch in training_data:
        src, tgt,labels = batch
        src_no_BOS_EOS = src[0][:,1:-1]
        tgt_no_BOS_EOS = tgt[0][:,1:-1]
        src_no_BOS_EOS[src[0][:,1:-1]==Constants.EOS] = Constants.PAD
        tgt_no_BOS_EOS[tgt[0][:,1:-1]==Constants.EOS] = Constants.PAD
        src_lengths = src_no_BOS_EOS.ne(Constants.PAD).sum(1)
        tgt_lengths = tgt_no_BOS_EOS.ne(Constants.PAD).sum(1)
        src_lengths_sorted,src_lengths_sorted_idx = src_lengths.sort(descending=True)
        tgt_lengths_sorted,tgt_lengths_sorted_idx = tgt_lengths.sort(descending=True)
        src_sorted = src_no_BOS_EOS[src_lengths_sorted_idx,:] #not include <s> on the source side
        tgt_sorted = tgt_no_BOS_EOS[tgt_lengths_sorted_idx,:]
        #tgt2src_lengths_sorted = tgt_lengths[src_lengths_sorted_idx]
        #sequence_mask function automatically moves the output to cuda if sequence_length is on cuda 
        mask_src = sequence_mask(sequence_length=src_lengths_sorted).float() # mask_src(batch, src_len+</s>) 
        mask_tgt = sequence_mask(sequence_length=tgt_lengths_sorted).float() # mask_tgt(batch, tgt_len+</s>)
        loss = trainBatch(src_batch=src_sorted, #src_batch(batch,src_len+</s>)
                          src_lengths = src_lengths_sorted, #src_lengths(batch,) this length counts src_len+</s>
                          src_lengths_sorted_idx = src_lengths_sorted_idx,
                          tgt_batch=tgt_sorted, #tgt_batch(batch,<s>+tgt_len+</s>) this includs both  <s> and </s> but only one is used for decoding (target size is one more than target length)
                          tgt_lengths = tgt_lengths_sorted, #tgt_lengths_sorted(batch,) this length counts tgt_len+</s>
                          tgt_lengths_sorted_idx = tgt_lengths_sorted_idx,
                          labels = labels,
                          mask_src = mask_src, # mask_src(batch, src_len+</s>)
                          mask_tgt = mask_tgt, # mask_tgt(batch, tgt_len+</s>)
                          encoder_src=encoder_src,
                          encoder_tgt=encoder_tgt,
                          classifier = classifier,
                          encoder_src_optimizer=encoder_src_optimizer,
                          encoder_tgt_optimizer=encoder_tgt_optimizer,
                          classifier_optimizer = classifier_optimizer,
                          criterion=criterion,
                          opt=opt)
         
        batch_num += 1
        total_loss += loss.item()
        
    return total_loss / batch_num

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-save_path', required=True)
    parser.add_argument('-save_every', type=int,default=5,help='save the model after this amount of epoch')
    parser.add_argument('-validate_every', type=int, default=1,help='compute loss, perplexity and accuracy on the validation data')
    parser.add_argument('-sample_every', type=int, default=5,help='translate a sample batch of validation data')
    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-learning_rate', type=float, default=.0001)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-embedding_size', type=int, default=512)
    parser.add_argument('-pretrained_embedding_src', default=None,help='use a pretrained embedding for source')
    parser.add_argument('-pretrained_embedding_tgt', default=None,help='use a pretrained embedding for target')
    parser.add_argument('-maxlen', type=int,default=50)
    parser.add_argument('-hidden_size', type=int, default=100)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-clip', type=float, default=0.0,help='gradient clipping helps prevent the exploding gradient problem') 
    parser.add_argument('-log', default=None)
    parser.add_argument('-weight_decay', type=float, default=0,help='L2 regularization')
    parser.add_argument('-attn_model', type=str,choices=['dot','general','concat','variational'],default='dot')
    parser.add_argument('-cuda', action='store_true',help='use cuda gpu, by default it is set to false.')
    parser.add_argument('-eps_sample_train', type=int, default=1,help='number of random samples for variational attention in the training phase')
    parser.add_argument('-eps_sample_eval', type=int, default=1,help='number of random samples for variational attention in the test phase')
    parser.add_argument('-var_hidden_size', type=int, default=1,help='hidden_size of variational network')

    opt = parser.parse_args()
    if torch.cuda.is_available() and opt.cuda:
        opt.cuda = True
    else:
        opt.cuda = False
    #========= Loading Dataset =========#
    data = torch.load(opt.data)
    opt.max_token_seq_len = data['settings'].max_token_seq_len

    #========= Preparing DataLoader =========#
    training_data = DataLoader(data['dict']['src'],
                               data['dict']['tgt'],
                               src_insts=data['train']['src'],
                               tgt_insts=data['train']['tgt'],
                               labels = data['train']['labels'],
                               batch_size=opt.batch_size,
                               cuda=opt.cuda)

    validation_data = DataLoader(data['dict']['src'],
                                 data['dict']['tgt'],
                                 src_insts=data['valid']['src'],
                                 tgt_insts=data['valid']['tgt'],
                                 labels = data['valid']['labels'],
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 test=True,
                                 cuda=opt.cuda)

    opt.src_vocab_size = training_data.src_vocab_size
    opt.tgt_vocab_size = training_data.tgt_vocab_size

    # all the (hyper-)parameters
    print(opt)

    #========= Preparing Model =========#
    # Initialize models
    encoder_src = EncoderRNN(opt.src_vocab_size,
                             opt.embedding_size,
                             opt.hidden_size,
                             opt,
                             pretrained_embedding=opt.pretrained_embedding_src)
    encoder_tgt = EncoderRNN(opt.tgt_vocab_size,
                             opt.embedding_size,
                             opt.hidden_size,
                             opt,
                             pretrained_embedding=opt.pretrained_embedding_tgt)

    #==== feed-forward classifier=====
    classifier = classifierMLP(4*opt.hidden_size,opt.hidden_size,2,dropout=opt.dropout)
    
    # Initialize optimizers and criterion
    #TODO:add learning rate picking strategies
    #optimizer = ScheduledOptim(optim.Adam(filter(lambda x: x.requires_grad, transformer.parameters()),betas=(0.9, 0.98), eps=1e-09),opt.d_model, opt.n_warmup_steps)
    encoder_src_optimizer = optim.Adam(encoder_src.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    encoder_tgt_optimizer = optim.Adam(encoder_tgt.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    # Move models to GPU
    if opt.cuda:
        encoder_src.cuda()
        encoder_tgt.cuda()
        classifier.cuda()
    
    best_model_acc = 0

    for epoch in range(opt.epoch):
        start_time = time.time()
        print('[Start training Epoch', epoch, ']')
        epoch_loss_avg = trainEpoch(training_data,encoder_src,encoder_tgt,classifier,encoder_src_optimizer,encoder_tgt_optimizer,classifier_optimizer,criterion,opt)

        print('Epoch: %s avg-training loss: %8.2f elapse-time: %3.3f' % (epoch,epoch_loss_avg,(time.time()-start_time)/60))

        checkpoint = {
            'encoder_src': encoder_src.state_dict(), #save state information of encoder 
            'encoder_tgt': encoder_tgt.state_dict(), #save state information of decoder
            'classifier': classifier.state_dict(), #save state information of decoder
            'settings': opt,
            'epoch': epoch}
        #save model 
        if epoch % opt.save_every==0:
            model_name = opt.save_path + '_epoch_{epoch:d}.chkpt'.format(epoch=epoch)
            torch.save(checkpoint, model_name)
        #validation
        if epoch % opt.validate_every==0:
            with torch.no_grad():
                encoder_src_eval = EncoderRNN(opt.src_vocab_size,
                                              opt.embedding_size,
                                              opt.hidden_size,
                                              opt,
                                              pretrained_embedding=opt.pretrained_embedding_src)
                encoder_tgt_eval = EncoderRNN(opt.tgt_vocab_size,
                                              opt.embedding_size,
                                              opt.hidden_size,
                                              opt,
                                              pretrained_embedding=opt.pretrained_embedding_tgt)
                classifier_eval = classifierMLP(4*opt.hidden_size,opt.hidden_size,2)
                encoder_src_eval.load_state_dict(checkpoint['encoder_src'])
                encoder_tgt_eval.load_state_dict(checkpoint['encoder_tgt'])
                classifier_eval.load_state_dict(checkpoint['classifier'])
                if opt.cuda:
                    encoder_src_eval.cuda()
                    encoder_tgt_eval.cuda()
                    classifier_eval.cuda()
                encoder_src_eval.eval()
                encoder_tgt_eval.eval()
                classifier_eval.eval()
                total_validate_loss = 0
                total_validate_correct = 0
                validate_batch_num = 0
                total_validation_predicted_words = 0
                all_pred_list = []
                all_labels_list = []
                validate_start = time.time()
                for batch in validation_data:
                    src, tgt,labels = batch
                    src_no_BOS_EOS = src[0][:,1:-1]
                    tgt_no_BOS_EOS = tgt[0][:,1:-1]
                    src_no_BOS_EOS[src[0][:,1:-1]==Constants.EOS] = Constants.PAD
                    tgt_no_BOS_EOS[tgt[0][:,1:-1]==Constants.EOS] = Constants.PAD
                    src_lengths = src_no_BOS_EOS.ne(Constants.PAD).sum(1)
                    tgt_lengths = tgt_no_BOS_EOS.ne(Constants.PAD).sum(1)
                    src_lengths_sorted,src_lengths_sorted_idx = src_lengths.sort(descending=True)
                    tgt_lengths_sorted,tgt_lengths_sorted_idx = tgt_lengths.sort(descending=True)
                    src_sorted = src_no_BOS_EOS[src_lengths_sorted_idx,:] #not include <s> on the source side
                    tgt_sorted = tgt_no_BOS_EOS[tgt_lengths_sorted_idx,:]
                    #sequence_mask function automatically moves the output to cuda if sequence_length is on cuda 
                    mask_src = sequence_mask(sequence_length=src_lengths_sorted).float() # mask_src(batch, src_len+</s>) 
                    mask_tgt = sequence_mask(sequence_length=tgt_lengths_sorted).float() # mask_tgt(batch, tgt_len+</s>)
                    enc_src_outputs, encoder_src_hidden = encoder_src_eval(src_sorted.transpose(0, 1),src_lengths_sorted) #
                    enc_tgt_outputs, encoder_tgt_hidden = encoder_tgt_eval(tgt_sorted.transpose(0, 1),tgt_lengths_sorted) #
                    #apply averaging here
                    enc_src_outputs = enc_src_outputs.sum(0,keepdim=True) / src_lengths_sorted.unsqueeze(1).expand(1,enc_src_outputs.size(1),enc_src_outputs.size(2)).float() #enc_src_outputs(1,batch,hidden_size)
                    enc_tgt_outputs = enc_tgt_outputs.sum(0,keepdim=True) / tgt_lengths_sorted.unsqueeze(1).expand(1,enc_tgt_outputs.size(1),enc_tgt_outputs.size(2)).float() #enc_tgt_outputs(1,batch,hidden_size)
                    #reorder to original 
                    enc_src_outputs = enc_src_outputs[:,src_lengths_sorted_idx.sort()[1],:].permute(1,0,2).squeeze(1) #enc_src_outputs(batch,hidden_size)
                    enc_tgt_outputs = enc_tgt_outputs[:,tgt_lengths_sorted_idx.sort()[1],:].permute(1,0,2).squeeze(1) #enc_tgt_outputs(batch,hidden_size)
                    
                    pred = classifier_eval(enc_src_outputs.clone(),enc_tgt_outputs.clone())
                    all_pred_list.append(pred.max(1)[1])
                    all_labels_list.append(labels)
                    #compute loss and accuracy here or after masking
                    #label_var_onehot = torch.zeros(pred.size(0),pred.size(1)).cuda() #debug
                    #for i in range(2): #debug
                    #    if i==0:
                    #        label_var_onehot[labels==0,i]=1 #debug
                    #    else:
                    #        label_var_onehot[labels==1,i]=1 #debug
                    
                    validate_batch_loss = criterion(pred.clone(),labels).item()
                    #validate_batch_loss = criterion(pred.clone(),label_var_onehot.clone()).item()
                    total_validate_correct += pred.max(1)[1].eq(labels).sum().item()
                    total_validation_predicted_words += pred.size(0) 
                    validate_batch_num += 1 
                    total_validate_loss += validate_batch_loss
                    
                valid_loss = total_validate_loss / validate_batch_num
                valid_accu = total_validate_correct / total_validation_predicted_words
                all_pred = torch.cat(all_pred_list)
                all_labels = torch.cat(all_labels_list)
                #[0] is precision, [1] is recall, [2] is F measure, [3] is the number of instances in each class not the correctly predicted instances
                accuracy = accuracy_score(all_labels,all_pred)
                precision = precision_recall_fscore_support(all_labels,all_pred, average=None)[0]
                recall = precision_recall_fscore_support(all_labels,all_pred, average=None)[1]
                samples_per_class = precision_recall_fscore_support(all_labels,all_pred, average=None)[3]
                F1_score = precision_recall_fscore_support(all_labels,all_pred, average=None)[2]
                F1_score_average_micro = precision_recall_fscore_support(all_labels,all_pred, average='micro')[2]
                F1_score_average_macro = precision_recall_fscore_support(all_labels,all_pred, average='macro')[2]
                if best_model_acc < accuracy:
                    best_model_acc = accuracy
                    model_name = opt.save_path + '_best.chkpt'
                    torch.save(checkpoint, model_name)
                    print('best model accuracy: ',accuracy)
                    all_pred_name = opt.save_path + '_best_all_preds.txt'
                    all_labels_name = opt.save_path + '_best_all_labels.txt'
                    with open(all_pred_name, 'w') as filehandle:
                        for listitem in all_pred:
                            filehandle.write('%s\n' % listitem.item())
                    with open(all_labels_name, 'w') as filehandle:
                        for listitem in all_labels:
                            filehandle.write('%s\n' % listitem.item())
                #confusion matrix: row true, column prediction
                conf_mat = confusion_matrix(all_labels,all_pred)
                print('  - (Validation) loss: {val_loss: 8.5f}, ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %,''elapse: {elapse:3.3f} min'.format(val_loss=valid_loss,ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu,elapse=(time.time()-validate_start)/60))
                print('accuracy: ',accuracy)
                print('precision: ',precision)
                print('recall: ',recall)
                print('number of instances per class in the ground truth: ',samples_per_class)
                print('F1-score: ',F1_score)
                print('F1-score_average_micro: ',F1_score_average_micro)
                print('F1-score_average_macro: ',F1_score_average_macro)
                print('confusion-matrix:\n',conf_mat)
                
   
if __name__ == '__main__':
    main()


