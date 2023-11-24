import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 1. Calculate Reconstruction Loss

def calculate_reconstruction_loss(decoder_mean, target, variance):
    """
    :param decoder_mean: batch_size (B) x number_of_features (F) from decoder output
    :param target: batch_size (B) x number_of_features (F) from data
    :param variance: log variance of decoder output = 0.
    :return: reconstruction loss
    """
    
    mu1 = decoder_mean
    mu2 = target
    neg_log_p = variance + torch.div(torch.pow(mu1 - mu2, 2), 2.0 * np.exp(2.0 * variance))

    return neg_log_p.sum() / (target.size(0)) # average over batch size

# 2. Calculate KL Divergence Loss

def calculate_kl_loss(z_0, z_T, z_q_mean, z_q_logvar, z_dims):
    log_p_z = log_Normal_standard(z_T, dim=z_dims)
    log_q_z = log_Normal_diag(z_0, z_q_mean, z_q_logvar, dim=z_dims)
    kl_loss = -(log_p_z - log_q_z)

    return kl_loss.sum() / (z_0.size(0))

def log_Normal_diag(x, mean, log_var, dim=None):
    log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) * torch.pow( torch.exp( log_var ), -1) )
    return torch.sum( log_normal, dim )

def log_Normal_standard(x, dim=None):
    log_normal = -0.5 * torch.pow( x , 2 )
    return torch.sum(log_normal, dim)

def kl_gaussian_sem(z_mean, z_logvar):
    kl_div = torch.exp(2*z_logvar) - 2*z_logvar + z_mean * z_mean
    kl_sum = kl_div.sum()
    return (kl_sum / z_mean.size(0))