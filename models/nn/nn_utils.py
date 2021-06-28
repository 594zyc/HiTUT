import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def drop_input(x, dropout_rate, training):
    if not training:
        return x
    # mask = x.bernoulli(1 - dropout_rate)
    # return x * mask
    mask = torch.ones_like(x[:, 0]).bernoulli(1 - dropout_rate)
    return x * mask.view(-1, 1).expand(x.shape)


def compute_mask(x, pad_idx=0):
    mask = torch.ne(x, pad_idx).float()
    if x.is_cuda:
        mask = mask.cuda()
    return mask


def to_one_hot(y_true, n_classes):
    # y_true: batch x time
    batch_size, length = y_true.size(0), y_true.size(1)
    y_onehot = torch.FloatTensor(batch_size, length, n_classes)  # batch x time x n_class
    if y_true.is_cuda:
        y_onehot = y_onehot.cuda()
    y_onehot.zero_()
    y_onehot = y_onehot.view(-1, y_onehot.size(-1))  # batch*time x n_class
    y_true = y_true.view(-1, 1)  # batch*time x 1
    y_onehot.scatter_(1, y_true, 1)  # batch*time x n_class
    return y_onehot.view(batch_size, length, -1)


def NegativeLogLoss(y_pred, y_true, mask=None, smoothing_eps=0.0):
    """
    Shape:
        - y_pred:    batch x time x vocab
        - y_true:    batch x time
        - mask:      batch x time
    """
    y_true_onehot = to_one_hot(y_true, y_pred.size(-1))  # batch x time x vocab
    if smoothing_eps > 0.0:
        y_true_onehot = y_true_onehot * (1.0 - smoothing_eps) + (1.0 - y_true_onehot) * smoothing_eps / (y_pred.size(-1) - 1)
    P = y_true_onehot * y_pred  # batch x time x vocab
    P = torch.sum(P, dim=-1)  # batch x time
    gt_zero = torch.gt(P, 0.0).float()  # batch x time
    epsilon = torch.le(P, 0.0).float() * 1e-8  # batch x time
    log_P = torch.log(P + epsilon) * gt_zero  # batch x time
    if mask is not None:
        log_P = log_P * mask
    output = -torch.sum(log_P, dim=1)  # batch
    return output


def NLL(y_pred, y_true_onehot, mask=None):
    """
    Shape:
        - y_pred:    batch x n_class
        - y_true:    batch x n_class
        - mask:      batch x n_class
    """
    y_true_onehot = y_true_onehot.float()
    P = y_true_onehot * y_pred  # batch x n_class
    gt_zero = torch.gt(P, 0.0).float()  # batch x n_class
    epsilon = torch.le(P, 0.0).float() * 1e-8  # batch x n_class
    log_P = torch.log(P + epsilon) * gt_zero  # batch x n_class
    if mask is not None:
        log_P = log_P * mask
    output = -torch.sum(log_P, dim=1)  # batch
    return torch.mean(output)


def masked_softmax(x, m=None, axis=-1):
    '''
    Softmax with mask (optional)
    '''
    x = torch.clamp(x, min=-15.0, max=15.0)
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-6)
    return softmax


def masked_mean(x, m=None, dim=1):
    """
        mean pooling when there're paddings
        input:  tensor: batch x time x h
                mask:   batch x time
        output: tensor: batch x h
    """
    if m is None:
        return torch.mean(x, dim=dim)
    x = x * m.unsqueeze(-1)
    mask_sum = torch.sum(m, dim=-1)  # batch
    tmp = torch.eq(mask_sum, 0).float()
    if x.is_cuda:
        tmp = tmp.cuda()
    mask_sum = mask_sum + tmp
    res = torch.sum(x, dim=dim)  # batch x h
    res = res / mask_sum.unsqueeze(-1)
    return res