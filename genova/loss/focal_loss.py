import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self,
    alpha: float = -1,
    gamma: float = 1,
    reduction: str = "mean") -> None:
        """_summary_

        Args:
            alpha (float, optional): class selection cofactor if set to -1 it won't work. Defaults to -1.
            gamma (float, optional): punish cofactor. Defaults to 1.
            reduction (str, optional): reduction type 'mean' 'sum' or 'none'. Defaults to "mean".
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction =='none':
            loss = loss
        else:
            raise TypeError('reduction should choose from \'mean\', \'sum\', \'none\'')

        return loss

class FocalOPCE(nn.Module):
    def __init__(self,
    alpha: float = -1,
    gamma: float = 1,
    eps: float = 1e-5,
    reduction: str = "mean") -> None:
        """_summary_

        Args:
            alpha (float, optional): class selection cofactor if set to -1 it won't work. Defaults to -1.
            gamma (float, optional): punish cofactor. Defaults to 1.
            reduction (str, optional): reduction type 'mean' 'sum' or 'none'. Defaults to "mean".
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        p = (inputs.softmax(-1)*targets.bool()).sum(-1)
        loss = -self.alpha*torch.pow(1-p,self.gamma)*(p + self.eps).log()
        if self.alpha >= 0: loss = self.alpha * loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction =='none':
            loss = loss
        else:
            raise TypeError('reduction should choose from \'mean\', \'sum\', \'none\'')

        return loss

class FocalKLDivLoss(nn.Module):
    def __init__(self,
    alpha: float = -1,
    gamma: float = 1,
    eps: float = 1e-5,
    reduction: str = "mean") -> None:
        """_summary_

        Args:
            alpha (float, optional): class selection cofactor if set to -1 it won't work. Defaults to -1.
            gamma (float, optional): punish cofactor. Defaults to 1.
            reduction (str, optional): reduction type 'mean' 'sum' or 'none'. Defaults to "mean".
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.klloss = nn.KLDivLoss(reduction='none')
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        inputs = inputs.softmax(-1)
        loss = self.klloss(inputs.log(),targets)
        p = (inputs*targets.bool()).sum(-1)
        loss = -torch.pow(1-p,self.gamma)*loss
        if self.alpha >= 0: loss = self.alpha * loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction =='none':
            loss = loss
        else:
            raise TypeError('reduction should choose from \'mean\', \'sum\', \'none\'')

        return loss
