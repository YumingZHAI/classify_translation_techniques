''' Handling the data io
author: Pooyan Safari'''

import argparse
import torch
import Constants
import pdb
import sys

def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1

            word_inst = words[:max_sent_len]

            if word_inst:
                # put all these elements in a list
                # each instance becomes e.g.: ['<s>', 'how', 'we', '</s>']
                # at the end, word_insts is a list of list
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    # here words are in string format
    return word_insts

def read_labels_from_file(inst_file):
    ''' Convert file into label lists '''
    
    labels = []
    with open(inst_file) as f:
        for label in f:
            label = label.split()
            labels += [label]
    print('[Info] Get {} instances from {}'.format(len(labels), inst_file))
          
    return labels

# if share vocabulary: regroup source and target words
def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurrence '''

    full_vocab = set(w for sent in word_insts for w in sent)

    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1    # increase the count

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:     # default value: 5
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    '''Word mapping to idx'''
    return [[word2idx[w] if w in word2idx else Constants.UNK for w in s] for s in word_insts]

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-train_labels', required=True)
    # valid -> is test data here
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-valid_labels', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_word_seq_len, opt.keep_case)
    train_labels = read_labels_from_file(opt.train_labels)
    
    if len(train_src_word_insts) != len(train_tgt_word_insts) != len(train_labels):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts),len(train_labels))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]
        train_labels = train_labels[:min_inst_count]
          
    # Remove empty instances
    train_src_word_insts, train_tgt_word_insts, train_labels = \
        list(zip(*[(s, t, l) for s, t, l in zip(train_src_word_insts, train_tgt_word_insts, train_labels) if s and t]))

    # Validation set
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tgt_word_insts = read_instances_from_file(
        opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)
    valid_labels = read_labels_from_file(opt.valid_labels)
    
    if len(valid_src_word_insts) != len(valid_tgt_word_insts) != len(valid_labels):
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts), len(valid_labels))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]
        # valid_labels = train_labels[:min_inst_count]  => error here before! 
        valid_labels = valid_labels[:min_inst_count]    

    # Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts, valid_labels = \
        list(zip(*[(s, t, l) for s, t, l in zip(valid_src_word_insts, valid_tgt_word_insts, valid_labels) if s and t]))

    # Build vocabulary
    if opt.vocab:
        # predefined_dicts = {'dict': {'src': src_word2idx,'tgt': tgt_word2idx}}
        # created by fasttext_embedding_tensor.py
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts,
            'labels':train_labels},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts,
            'labels':valid_labels}
    }

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
