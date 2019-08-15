'''
extract fasttext word embedding tensors from fasttext word embeddings and build the predefined dictionary for the data
Python 3.5.5 :: Anaconda, Inc. 
'''

import numpy
import torch
import nwa.Constants as Constants

src_emb_txt = '../dataset/en-fr-corpus/fasttext-normalized/english-normalized.vec'
tgt_emb_txt = '../dataset/en-fr-corpus/fasttext-normalized/french-normalized.vec'
dict_path = '../dataset/en-fr-corpus/fasttext-normalized/predefined_dicts.fr-en.pt'
src_emb_tensor = '../dataset/en-fr-corpus/fasttext-normalized/embedding_en.pt'
tgt_emb_tensor = '../dataset/en-fr-corpus/fasttext-normalized/embedding_fr.pt'

with open(src_emb_txt, 'r') as src_txt, open(tgt_emb_txt, 'r') as tgt_txt:
    next(src_txt)
    next(tgt_txt)
    src_word2emb = {}
    tgt_word2emb = {}
    for line in src_txt:
        if line.split()[0] == '</s>':  # end of sentence
            src_EOS_emb = numpy.array(line.split()[1:]).astype(numpy.float)
        else:
            src_word2emb[line.split()[0]] = numpy.array(line.split()[1:]).astype(numpy.float)
    for line in tgt_txt:
        if line.split()[0] == '</s>':
            tgt_EOS_emb = numpy.array(line.split()[1:]).astype(numpy.float)
        else:
            tgt_word2emb[line.split()[0]] = numpy.array(line.split()[1:]).astype(numpy.float)

src_word2idx = {}
tgt_word2idx = {}
#build predefined dictionary
#PAD=0<blank>, UNK=1,BOS=2<s>,EOS=3</s> and also whitespace character to the dictionary take character embedding from already trained models
src_word2idx[Constants.BOS_WORD] = Constants.BOS
src_word2idx[Constants.EOS_WORD] = Constants.EOS
src_word2idx[Constants.PAD_WORD] = Constants.PAD
src_word2idx[Constants.UNK_WORD] = Constants.UNK
#{Constants.BOS_WORD: Constants.BOS,Constants.EOS_WORD: Constants.EOS, Constants.PAD_WORD: Constants.PAD,Constants.UNK_WORD: Constants.UNK,' ':4}
tgt_word2idx[Constants.BOS_WORD] = Constants.BOS
tgt_word2idx[Constants.EOS_WORD] = Constants.EOS
tgt_word2idx[Constants.PAD_WORD] = Constants.PAD
tgt_word2idx[Constants.UNK_WORD] = Constants.UNK
#tgt_word2idx{Constants.BOS_WORD: Constants.BOS,Constants.EOS_WORD: Constants.EOS, Constants.PAD_WORD: Constants.PAD,Constants.UNK_WORD: Constants.UNK,' ':4}
for w in src_word2emb:
    src_word2idx[w] = list(src_word2emb).index(w)+4
for w in tgt_word2emb:
    tgt_word2idx[w] = list(tgt_word2emb).index(w)+4

predefined_dicts = {'dict': {'src': src_word2idx,'tgt': tgt_word2idx}}
#build pytorch tensor for embeddings
#add PAD=0, UNK=1,BOS=2<s>,EOS=3</s> and also whitespace character to embedding matrix
src_embedding_tensor = numpy.zeros((len(src_word2idx),src_EOS_emb.shape[0])) 
tgt_embedding_tensor = numpy.zeros((len(tgt_word2idx),tgt_EOS_emb.shape[0]))
for w in src_word2emb:
    src_embedding_tensor[src_word2idx[w]] = src_word2emb[w]
for w in tgt_word2emb:
    tgt_embedding_tensor[tgt_word2idx[w]] = tgt_word2emb[w]

#add the embedding of the eos
src_embedding_tensor[Constants.EOS] = src_EOS_emb
tgt_embedding_tensor[Constants.EOS] = tgt_EOS_emb
src_embedding_tensor = torch.FloatTensor(src_embedding_tensor)
tgt_embedding_tensor = torch.FloatTensor(tgt_embedding_tensor)

torch.save(src_embedding_tensor,src_emb_tensor)
torch.save(tgt_embedding_tensor,tgt_emb_tensor)
torch.save(predefined_dicts,dict_path)
