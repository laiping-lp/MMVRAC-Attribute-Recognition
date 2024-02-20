import random
import torch
import torch.nn as nn


class instancenorm_1d(nn.Module):
    def __init__(self, num_features, affine=False, eps=1e-6, track_running_stat=False) -> None:
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.track_running_stat = track_running_stat

    def forward(self, x):
        if not self.training: ##### important
            return x
        if random.random() > 0.5: ##### important
            return x

        assert x.dim() == 3
        B,N,C = x.shape
        assert C == self.num_features
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig
        return x_normed

class layernorm_1d(nn.Module):
    def __init__(self, num_features, affine=False, eps=1e-6, track_running_stat=False) -> None:
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.track_running_stat = track_running_stat

    def forward(self, x):
        if not self.training: ##### important
            return x
        if random.random() > 0.5: ##### important
            return x
        
        assert x.dim() == 3
        B,N,C = x.shape
        assert C == self.num_features
        eps = self.eps
        mean = x.mean(dim=2, keepdim=True).detach()
        var = x.var(dim=2, keepdim=True).detach()
        x_new = (x-mean) / (var + eps).sqrt()
        return x_new
    
class batchnorm_1d(nn.Module):
    def __init__(self, num_features, affine=False, eps=1e-6, track_running_stat=False) -> None:
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.track_running_stat = track_running_stat

    def forward(self, x):
        if not self.training: ##### important
            return x
        if random.random() > 0.5: ##### important
            return x
        
        assert x.dim() == 3
        B,N,C = x.shape
        assert C == self.num_features
        eps = self.eps
        mean = x.mean(dim=0, keepdim=True).detach()
        var = x.var(dim=0, keepdim=True).detach()
        x_new = (x-mean) / (var + eps).sqrt()
        return x_new