'''
extract polyglot character embeddings from polyglot word embeddings
author: Pooyan Safari
'''

# run this script under python 2.7
# need to add empty file __init__.py under cnn/nwa/ to import nwa.Constants
import torch
import pickle
import numpy
import nwa.Constants as Constants
import sys

word_embedding_src = '../dataset/polyglot_embeddings/polyglot-en.pkl'
word_embedding_tgt = '../dataset/polyglot_embeddings/polyglot-fr.pkl'
dict_path = '../dataset/polyglot_embeddings/predefined_dicts.char.fr-en.pt'
emb_tensor_src = '../dataset/polyglot_embeddings/embedding_src.pt'
emb_tensor_tgt = '../dataset/polyglot_embeddings/embedding_tgt.pt'

#read polyglot pickle files
words_src, embeddings_src = pickle.load(open(word_embedding_src, 'rb'))
words_tgt, embeddings_tgt = pickle.load(open(word_embedding_tgt, 'rb'))

#extract character embeddings
vectors_src = {}
# 0-3: (u'<UNK>', u'<S>', u'</S>', u'<PAD>')
for word in words_src[4:]:
    # dimension 64
    vec = embeddings_src[words_src.index(word),:]
    for char in word.lower():
        if char in vectors_src:
            # embedding vector sum, keep dimension 64
            # increase the count
            vectors_src[char] = (vectors_src[char][0] + vec,vectors_src[char][1] + 1)
        else:
            vectors_src[char] = (vec, 1)

avg_vector_src = {}
for char in vectors_src:
    # sum the word embedding where the character appears and divide by its number of appearance
    avg_vector_src[char] = vectors_src[char][0] / vectors_src[char][1]

# for word in avg_vector_src.keys():
#     if not word:
#         print('empty string found!')
#     if word.isspace():
#         print('white space string found!')
#     try:
#         float(word)
#         print('number is detected! ',word)
#     except:
#         continue

vectors_tgt = {}
for word in words_tgt[4:]:
    vec = embeddings_tgt[words_tgt.index(word),:]
    for char in word.lower(): 
        if char in vectors_tgt:
            vectors_tgt[char] = (vectors_tgt[char][0] + vec,vectors_tgt[char][1] + 1)
        else:
            vectors_tgt[char] = (vec, 1)

avg_vector_tgt = {}
for char in vectors_tgt:
    avg_vector_tgt[char] = vectors_tgt[char][0] / vectors_tgt[char][1]

src_word2idx = {}
tgt_word2idx = {}
#build predefined dictionary
#PAD=0<blank>, UNK=1, BOS=2<s>, EOS=3</s> and also whitespace character to the dictionary
src_word2idx[Constants.BOS_WORD] = Constants.BOS
src_word2idx[Constants.EOS_WORD] = Constants.EOS
src_word2idx[Constants.PAD_WORD] = Constants.PAD
src_word2idx[Constants.UNK_WORD] = Constants.UNK
src_word2idx[' '] = 4

tgt_word2idx[Constants.BOS_WORD] = Constants.BOS
tgt_word2idx[Constants.EOS_WORD] = Constants.EOS
tgt_word2idx[Constants.PAD_WORD] = Constants.PAD
tgt_word2idx[Constants.UNK_WORD] = Constants.UNK
tgt_word2idx[' '] = 4

for w in avg_vector_src.keys():
    src_word2idx[w] = avg_vector_src.keys().index(w)+5
for w in avg_vector_tgt.keys():
    tgt_word2idx[w] = avg_vector_tgt.keys().index(w)+5

predefined_dicts = {'dict': {'src': src_word2idx,'tgt': tgt_word2idx}}
torch.save(predefined_dicts,dict_path)

#build pytorch tensor for embeddings
#add PAD=0, UNK=1,BOS=2<s>,EOS=3</s> and also whitespace character to embedding matrix
src_embedding_tensor = numpy.zeros((len(vectors_src)+5,embeddings_src.shape[1])) 
tgt_embedding_tensor = numpy.zeros((len(vectors_tgt)+5,embeddings_tgt.shape[1]))

for w in avg_vector_src.keys():
    src_embedding_tensor[src_word2idx[w]] = avg_vector_src[w]
for w in avg_vector_tgt.keys():
    tgt_embedding_tensor[tgt_word2idx[w]] = avg_vector_tgt[w]

#get embedding of the UNK from word embedding to character embedding
# embeddings_src: 0-3: (u'<UNK>', u'<S>', u'</S>', u'<PAD>')
src_embedding_tensor[1] = embeddings_src[0]
tgt_embedding_tensor[1] = embeddings_tgt[0]
#TODO:take values for eos and bos embeddings and paddings from embeddings_src embeddings_tgt
src_embedding_tensor = torch.FloatTensor(src_embedding_tensor)
tgt_embedding_tensor = torch.FloatTensor(tgt_embedding_tensor)
torch.save(src_embedding_tensor,emb_tensor_src)
torch.save(tgt_embedding_tensor,emb_tensor_tgt)


