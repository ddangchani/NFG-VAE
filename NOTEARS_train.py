import time
import datetime
import pickle
import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset
import math
import numpy as np
from utils import *
from model import *
from data import *
from loss import *
from tqdm import tqdm

from NOTEARS import notears_linear

# 1. Arguments(argparse)

parser = argparse.ArgumentParser(description='NOTEARS Algorithm')

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
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--x_dims', type=int, default=1,
                    help='the dimension of each variable')


# 1.2. Arguments for NOTEARS
parser.add_argument('--lambda1', type=float, default=0.1,
                    help='l1 penalty parameter')
parser.add_argument('--loss_type', type=str, default='l2',
                    help='loss type')
parser.add_argument('--max_iter', type=int, default=100,
                    help='max num of dual ascent steps')
parser.add_argument('--h_tol', type=float, default=1e-8,
                    help='exit if |h(w_est)| <= htol')
parser.add_argument('--rho_max', type=float, default=1e+16,
                    help='exit if rho >= rho_max')
parser.add_argument('--w_threshold', type=float, default=0.3,
                    help='drop edge if |weight| < threshold')


args = parser.parse_args()
args.z_size = args.node_size # the number of latent variables
print(args)

# Device configuration (GPU or MPS or CPU)

# Seed
np.random.seed(args.seed)

# Folder to save the results, models, dataset

now = datetime.datetime.now().strftime('%m%d_%H%M')
if args.dependence_type == 1:
    folder = f'results/dependence/{now}_NOTEARS_node{args.node_size}_prop{int(args.dependence_prop*100)}'
else:
    folder = f'results/independence/{args.graph_dist}/{now}_NOTEARS_node{args.node_size}'

if not os.path.exists(folder):
    os.makedirs(folder)

# Save the arguments
meta_file = os.path.join(folder, 'meta.pkl')
log_file = os.path.join(folder, 'log.txt')
log = open(log_file, 'w')

pickle.dump(args, open(meta_file, 'wb'))

# 2. Load Data

# 2.1. Generate DAG
G = generate_random_dag(d = args.node_size, degree=args.graph_degree, seed=args.seed)

# 2.2. Generate Data
if args.dependence_type == 1:
    X, cov, cov_prev, G = generate_linear_sem_correlated(G, args.data_sample_size, args.dependence_prop, args.seed, return_cov=True, x_dims=args.x_dims, return_graph=True)
else:
    X = generate_linear_sem(graph=G, n=args.data_sample_size, dist=args.graph_dist, linear_type=args.graph_linear_type, loc=args.graph_mean, scale=args.graph_scale, seed=args.seed, x_dims=args.x_dims)

X = X.squeeze(-1)

# save X to file
data_file = os.path.join(folder, 'data.pkl')
pickle.dump(X, open(data_file, 'wb'))

# save covariance matrix to file
if args.dependence_type == 1:
    cov_file = os.path.join(folder, 'cov.pkl')
    pickle.dump(cov, open(cov_file, 'wb'))
    cov_prev_file = os.path.join(folder, 'cov_prev.pkl')
    pickle.dump(cov_prev, open(cov_prev_file, 'wb'))

# 3. Train the model

pbar = tqdm(total=args.max_iter)
W_est = notears_linear(X, args.lambda1, args.loss_type, args.max_iter, args.h_tol, args.rho_max, args.w_threshold, pbar=pbar)

# 4. Save the results

# Save the Graph metrics
fdr, tpr, fpr, shd, nnz = count_accuracy(G, nx.DiGraph(W_est))

graph = W_est.copy()
graph[np.abs(graph) < 0.1] = 0
# print(graph)
fdr, tpr, fpr, shd, nnz = count_accuracy(G, nx.DiGraph(graph))
print('threshold 0.1, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz, file=log)

graph[np.abs(graph) < 0.2] = 0
# print(graph)
fdr, tpr, fpr, shd, nnz = count_accuracy(G, nx.DiGraph(graph))
print('threshold 0.2, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz, file=log)

graph[np.abs(graph) < 0.3] = 0
# print(graph)
fdr, tpr, fpr, shd, nnz = count_accuracy(G, nx.DiGraph(graph))
print('threshold 0.3, Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz, file=log)

f = open(folder + '/trueG.txt', 'w')
matG = np.matrix(nx.to_numpy_array(G))
for line in matG:
    np.savetxt(f, line, fmt='%.5f')
f.closed

f1 = open(folder + '/predG.txt', 'w')
for line in W_est:
    np.savetxt(f1, line, fmt='%.5f')
f1.closed

if log is not None:
    print(folder)
    log.close()

print('Done!')