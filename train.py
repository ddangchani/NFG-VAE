import time
import datetime
import pickle
import os
import argparse

import torch
from torch.optim import lr_scheduler
import math
import numpy as np
from utils import *
from model import *
from data import *


# 1. Arguments(argparse)

parser = argparse.ArgumentParser(description='PyTorch Training')

# 1.1. Arguments for Data

parser.add_argument('--data_sample_size', type=int, default=5000, 
                    help='the number of samples of data')
parser.add_argument('--node_size', type=int, default=10, 
                    help='the number of nodes')
parser.add_argument('--graph_degree', type=int, default=2, 
                    help='the expected degree of random graph')
parser.add_argument('--graph_dist', type=str, default='normal', 
                    help='the distribution of random graph')
parser.add_argument('--graph_scale', type=float, default=1.0, 
                    help='the scale parameter of distribution')
parser.add_argument('--graph_mean', type=float, default=0.0, 
                    help='the mean parameter of distribution')
parser.add_argument('--graph_linear_type', type=str, default='linear', 
                    help='the type of linear graph')
parser.add_argument('--dependence_type', type=int, default=0, 
                    help='Dependent noise distribution or not')
parser.add_argument('--dependence_prop', type=float, default=0.5, 
                    help='the proportion of dependent noise')

# 1.2. Arguments for Hyperparameters
parser.add_argument('--optimizer', type = str, default = 'Adam',
                    help = 'the choice of optimizer used')
parser.add_argument('--graph_threshold', type=  float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                    help = 'threshold for learned adjacency matrix binarization')
parser.add_argument('--tau_A', type = float, default=0.0,
                    help='coefficient for L-1 norm of A.')
parser.add_argument('--lambda_A',  type = float, default= 0.,
                    help='coefficient for DAG constraint h(A).')
parser.add_argument('--c_A',  type = float, default= 1,
                    help='coefficient for absolute value h(A).')
parser.add_argument('--use_A_connect_loss',  type = int, default= 0,
                    help='flag to use A connect loss')
parser.add_argument('--use_A_positiver_loss', type = int, default = 0,
                    help = 'flag to enforce A must have positive values')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default= 300,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default = 100, # note: should be divisible by sample size, otherwise throw an error
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=3e-3,  # basline rate = 1e-3
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--k_max_iter', type = int, default = 1e2,
                    help ='the max iteration number for searching lambda and c')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')


args = parser.parse_args()

# Device configuration (GPU or MPS or CPU)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Seed
torch.manual_seed(args.seed)

# Folder to save the results, models, dataset

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
if args.dependece_type == 1:
    folder = f'results/dependence/{args.dependence_prop * 100}%_dependence/{now}'
else:
    folder = f'results/independence/{args.graph_dist}/{now}'

if not os.path.exists(folder):
    os.makedirs(folder)

# Save the arguments
meta_file = os.path.join(folder, 'meta.pkl')
encoder_file = os.path.join(folder, 'encoder.pt')
decoder_file = os.path.join(folder, 'decoder.pt')

log_file = os.path.join(folder, 'log.txt')
log = open(log_file, 'w')

pickle.dump(args, open(meta_file, 'wb'))

# 2. Data Loading






# 3. Model Loading






# 4. Training Loop (epoch, batch, loss, optimizer, etc.)






# 5. Save Model (checkpoint, etc.)