import torch
import torch.nn as nn
import numpy as np

class FocalLoss(object):
    def __init__(self, sampler=None, batch_size=None, training=False):
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.training = training
        if self.training: self.p = torch.Tensor(np.concatenate([1/(sampler.dataset.capacity*sampler.p),np.ones(batch_size)]))
    def __call__(self, outputs, label, label_mask=None, idx=None, local_rank=None, reduction='mean'):
        focal_weight = (1-outputs.sigmoid())*label+outputs.sigmoid()*(1-label)
        if self.training: 
            priority_weight = self.p[idx].view(-1,1,1).to(local_rank)
            loss = self.loss_fn(outputs,label)*focal_weight*priority_weight
        else:
            loss = self.loss_fn(outputs,label)*focal_weight

        if reduction=='mean': 
            if label_mask!=None: loss = loss[label_mask]
            return loss.sum(-1).mean()
        elif reduction=='sum':
            if label_mask!=None: loss = loss[label_mask]
            return loss.sum()
        elif reduction=='max':
            loss = loss.sum(-1)
            if label_mask!=None: loss = loss.masked_fill(~label_mask, 0)
            loss,_ = loss.max(-1)
            return loss
        else:
            raise NotImplementedError