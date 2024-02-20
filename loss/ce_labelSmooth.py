import torch
import torch.nn as nn
from torch.nn import functional as F
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    


###### XDED loss
class xded_loss(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, bs=64, num_instance=4, tao=1.0, lam=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.bs = bs
        self.num_ins = num_instance
        self.tao = tao
        self.lam = lam

    def forward(self, inputs, targets):
        bs = self.bs
        tao = self.tao
        lam = self.lam
        num_ins = self.num_ins
        classes = self.num_classes

        
        log_probs = self.logsoftmax(inputs)
        ####### XDED
        ####### score: (batch_size, classes)
        probs = nn.Softmax(dim=1)(inputs / tao)
        probs_mean = probs.reshape(bs//num_ins,num_ins,classes).mean(1,True)

        probs_xded = probs_mean.repeat(1,num_ins,1).view(-1,classes).detach()
        # log_probs_xded = torch.log(probs_xded)

        # log_probs_mean = log_probs.reshape(bs//num_ins,num_ins,classes).mean(1,True)
        # log_probs_xded = log_probs_mean.repeat(1,num_ins,1).view(-1,classes)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / classes
        loss = (- targets * log_probs).mean(0).sum()
        loss_xded = (- probs_xded * log_probs).mean(0).sum()
        return loss + lam * loss_xded