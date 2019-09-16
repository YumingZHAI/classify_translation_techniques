import torch
from torch.nn import functional
from torch.autograd import Variable
import Constants
import pdb

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.range(0, max_len - 1).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand)).long()
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
        seq_length_expand = seq_length_expand.cuda()
    return seq_range_expand < seq_length_expand

def masked_cross_entropy(logits, target, length,use_cuda=True):
    if use_cuda:
        length = Variable(torch.cuda.LongTensor(length))
    else:
        length = Variable(torch.LongTensor(length))
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    logits_flat = logits.view(-1, logits.size(-1)) # logits_flat(batch*max_len,tgt_vocab_size)
    log_probs_flat = functional.log_softmax(logits_flat) # log_probs_flat(batch*max_len,tgt_vocab_size)
    target_flat = target.view(-1, 1) # target_flat(batch*max_len,1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat) # losses_flat(batch*max_len,1)
    losses = losses_flat.view(*target.size()) # losses(batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1)) # mask(batch, max_len)
    losses = losses * mask.float() #losses(batch,tgt_len+</s>)
    loss = losses.sum() / length.float().sum()
    return loss

def masked_validation_cross_entropy(logits, target, length,use_cuda=True):
    if use_cuda:
        length = Variable(torch.cuda.LongTensor(length))
    else:
        length = Variable(torch.LongTensor(length))
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    batch_size = logits.size(0)
    max_len = logits.size(1)
    target_padded = torch.zeros((batch_size,max_len)).long()
    target_padded[:,:target.size(1)]=target
    if use_cuda:
        target_padded = target_padded.cuda()
    logits_flat = logits.view(-1, logits.size(-1)) # logits_flat(batch*max_len,tgt_vocab_size)
    #why log_softmax on all the values I think masking should be held here
    log_probs_flat = functional.log_softmax(logits_flat) # log_probs_flat(batch*max_len,tgt_vocab_size)
    target_flat = target_padded.view(-1, 1) #target_flat(batch*max_len,1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat) # losses_flat(batch*max_len,1)
    losses = losses_flat.view(*target_padded.size()) # losses(batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target_padded.size(1)) # mask(batch, max_len)
    losses = losses * mask.float() #losses(batch,tgt_len+</s>)
    loss = losses.sum() / length.float().sum()
    #number of correct predicted words
    total_batch_correct = logits_flat.max(1)[1].eq(target_flat.squeeze(1))
    total_batch_correct = (total_batch_correct.view(*target_padded.size()) * mask).sum()
    return loss,total_batch_correct

#def masked_validation_cross_entropy(logits, target, length,tgt_lengths_sorted,use_cuda=True):
#    if use_cuda:
#        length = Variable(torch.cuda.LongTensor(length))
#    else:
#        length = Variable(torch.LongTensor(length))
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
#    batch_size = logits.size(0)
#    max_len = logits.size(1)
#    target_padded = torch.zeros((batch_size,max_len)).long()
#    target_padded[:,:target.size(1)]=target
#    if use_cuda:
#        target_padded = target_padded.cuda()
#    logits_flat = logits.view(-1, logits.size(-1)) # logits_flat(batch*max_len,tgt_vocab_size)
    #why log_softmax on all the values I think masking should be held here
#    log_probs_flat = functional.log_softmax(logits_flat) # log_probs_flat(batch*max_len,tgt_vocab_size)
#    target_flat = target_padded.view(-1, 1) #target_flat(batch*max_len,1)
#    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat) # losses_flat(batch*max_len,1)
#    losses = losses_flat.view(*target_padded.size()) # losses(batch, max_len)
#    mask = sequence_mask(sequence_length=tgt_lengths_sorted.max(length.float()), max_len=target_padded.size(1)) # mask(batch, max_len)
#    losses = losses * mask.float() #losses(batch,tgt_len+</s>)
#    loss = losses.sum() / length.float().sum()
#    pdb.set_trace()
    #number of correct predicted words
#    total_batch_correct = logits_flat.max(1)[1].eq(target_flat.squeeze(1))
#    total_batch_correct = (total_batch_correct.view(*target_padded.size()) * mask).sum()
#    return loss,total_batch_correct
