import time
import datetime
import pickle
import os
import argparse

import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset
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
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default= 1.0,
                    help='LR decay factor.')


args = parser.parse_args()
print(args.dependence_type)

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
if args.dependence_type == 1:
    folder = f'results/dependence/{now}'
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

# 2. Load Data

# 2.1. Generate DAG
G = generate_random_dag(d = args.node_size, degree=args.graph_degree, seed=args.seed)

# 2.2. Generate Data
if args.dependence_type == 1:
    X = generate_linear_sem_correlated(G, args.data_sample_size, args.dependence_prop, args.seed)
else:
    X = generate_linear_sem(graph=G, n=args.data_sample_size, dist=args.graph_dist, linear_type=args.graph_linear_type, loc=args.graph_mean, scale=args.graph_scale, seed=args.seed)

# save X to file
data_file = os.path.join(folder, 'data.pkl')
pickle.dump(X, open(data_file, 'wb'))


# 2.3. To Pytorch Tensor(VAE)

feat_train = torch.FloatTensor(X).to(device)
feat_valid = torch.FloatTensor(X).to(device)
feat_test = torch.FloatTensor(X).to(device)

train_data = TensorDataset(feat_train, feat_train)
valid_data = TensorDataset(feat_valid, feat_valid)
test_data = TensorDataset(feat_test, feat_train)

train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size)
test_data_loader = DataLoader(test_data, batch_size=args.batch_size)


# 3. Load Model

off_diag = np.ones([args.node_size, args.node_size]) - np.eye(args.node_size)
rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
rel_rec = torch.DoubleTensor(rel_rec).to(device)
rel_send = torch.DoubleTensor(rel_send).to(device)
triu_indices = get_triu_indices(args.node_size).to(device)
tril_indices = get_tril_indices(args.node_size).to(device)

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)

# Adjacency Matrix
adj_A = np.zeros([args.node_size, args.node_size])

# 3.1. Encoder and Decoder
encoder = Encoder().to(device)
decoder = Decoder().to(device)

# 3.2. Flow Layer
flow = FlowLayer().to(device)

# 3.3. Optimizer
if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(flow.parameters()), lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(list(encoder.parameters()) + list(decoder.parameters()) + list(flow.parameters()), lr=args.lr, momentum=0.9, nesterov=True)
elif args.optimizer == 'LBFGS':
    optimizer = torch.optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()) + list(flow.parameters()), lr=args.lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None)
else:
    raise ValueError('Optimizer not recognized.')

# 3.4. Learning Rate Scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

prox_plus = torch.nn.Threshold(0.,0.)

def stau(w, tau):
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1

def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr

# 4. Training Loop (epoch, batch, loss, optimizer, etc.)

def train():

    return ...





# 5. Save Model (checkpoint, etc.)