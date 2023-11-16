import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 1. Calculate Reconstruction Loss

def calculate_reconstruction_loss(preds, target, variance):
    mu1 = preds
    mu2 = target
    neg_log_p = variance + torch.div(torch.pow(mu1-mu2, 2), 2.0 * np.exp(2.0 * variance))

    return neg_log_p.sum() / (target.size(0))

# 2. Calculate KL Divergence Loss

def calculate_kl_loss(preds, zsize):
    predsnew = preds.squeeze(1)
    mu = predsnew[:,0:zsize]
    log_sigma = predsnew[:,zsize:2*zsize]
    kl_div = torch.exp(2*log_sigma) - 2*log_sigma + mu * mu
    kl_sum = kl_div.sum()

    return (kl_sum / (preds.size(0)) - zsize)*0.5