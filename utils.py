import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


# 1. Metric functions

def count_accuracy(G_true : nx.DiGraph, G_est : nx.DiGraph, G_und : nx.DiGraph = None):
    """
    Count accuracy of estimated graph G_est with respect to true graph G_true.
    If G_und is not None, then the accuracy is counted for undirected graphs.

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse (Structural Hamming Distance)
        pred_size: number of predicted edges
    """

    A_true = nx.to_numpy_array(G_true) # true adjacency matrix
    A_est = nx.to_numpy_array(G_est) # estimated adjacency matrix
    A_und = None if G_und is None else nx.to_numpy_array(G_und) # undirected adjacency matrix
    d = A_true.shape[0] # number of nodes

    # Linear indices of non-zero elements
    if A_und is not None:
        pred_und = np.flatnonzero(A_und)
    pred = np.flatnonzero(A_est)
    true = np.flatnonzero(A_true)
    true_reversed = np.flatnonzero(A_true.T)
    true_skeleton = np.concatenate((true, true_reversed))

    # Count True Positive (# of correctly predicted edges)
    tp = np.intersect1d(pred, true, assume_unique=True)
    if A_und is not None:
        tp_und = np.intersect1d(pred_und, true_skeleton, assume_unique=True)
        tp = np.concatenate((tp, tp_und))
    
    # False Positive (# of incorrectly predicted edges)
    fp = np.setdiff1d(pred, true_skeleton, assume_unique=True)
    if A_und is not None:
        fp_und = np.setdiff1d(pred_und, true_skeleton, assume_unique=True)
        fp = np.concatenate((fp, fp_und))

    # Reverse
    extra = np.setdiff1d(true, pred, assume_unique=True)
    reverse = np.intersect1d(extra, true_reversed, assume_unique=True)

    #Compute Ratio
    pred_size = len(pred)
    if A_und is not None:
        pred_size += len(pred_und)
    true_neg_size = 0.5 * d * (d - 1) - len(true)

    fdr = float(len(reverse) + len(fp)) / max(pred_size, 1)
    tpr = float(len(tp)) / max(len(true), 1)
    fpr = float(len(reverse) + len(fp)) / max(true_neg_size, 1)

    # SHD
    A_lower = np.tril(A_est + A_est.T)
    if A_und is not None:
        A_lower += np.tril(A_und + A_und.T)
    
    pred_lower = np.flatnonzero(A_lower)
    true_lower = np.flatnonzero(np.tril(A_true + A_true.T))
    extra_lower = np.setdiff1d(pred_lower, true_lower, assume_unique=True)
    missing_lower = np.setdiff1d(true_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    return fdr, tpr, fpr, shd, pred_size

# 2. Some useful functions for modeling