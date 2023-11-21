import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 1. Calculate Reconstruction Loss

def calculate_reconstruction_loss(preds, target, variance):
    mu1 = preds
    mu2 = target
    neg_log_p = variance + torch.div(torch.pow(mu1-mu2, 2), 2.0 * np.exp(2.0 * variance))

    return neg_log_p.sum() / (target.size(0)) # average over batch size

# 2. Calculate KL Divergence Loss

def calculate_kl_loss(z_0, z_1, z_q_mean, z_q_logvar):
    log_p_z = log_Normal_standard(z_1, dim=1)
    log_q_z = log_Normal_diag(z_0, z_q_mean, z_q_logvar, dim=1)
    kl_loss = -(log_p_z - log_q_z)

    return kl_loss.sum() / (z_0.size(0))


def log_Normal_diag(x, mean, log_var, average=False, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) * torch.pow( torch.exp( log_var ), -1) )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )

def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow( x , 2 )
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)

def kl_gaussian_sem(logits):
    mu = logits
    kl_div = mu * mu
    kl_sum = kl_div.sum()
    return (kl_sum / (logits.size(0)))*0.5