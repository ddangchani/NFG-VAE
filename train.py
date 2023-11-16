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
from loss import calculate_reconstruction_loss, calculate_kl_loss


from __future__ import print_function

from utils.distributions import log_Normal_diag, log_Normal_standard, log_Bernoulli

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
parser.add_argument('--cuda', type=int, default=0,
                    help='use cuda or not')
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
vae_indep_file = os.path.join(folder, 'vae_indep.pt')
vae_dep_file = os.path.join(folder, 'vae_dep.pt')

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

# 3.1. Load VAE

vae = VAE(args=args)

# 3.3. Optimizer
optimizer = torch.optim.Adam(list(vae.parameters()), lr=args.lr)

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
    '''
    FOR INDEPENDENT NOISE CASE (Augmented Lagrangian Method)
    related LR to c_A, whenever c_A gets big, reduce LR proportionally
    '''
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


def train(epoch, model, best_val_loss, ground_truth_G, lambda_A, c_A, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []
    
    # set model in training mode
    vae.train()

    z = {}

    # start training
    if args.warmup == 0:
        beta = 1.
    else:
        beta = 1.* (epoch-1) / args.warmup
        if beta > 1.:
            beta = 1.
    print('beta: {}'.format(beta))

    for batch_idx, (data, relations) in enumerate(train_loader):

        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data).double(), Variable(relations).double()

        # reshape data
        relations = relations.unsqueeze(2)

        optimizer.zero_grad()

        # myA = encoder.adj_A, adj_A_tilt is identity matrix -> 왜 필요한가?
        z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, output, x_mean, x_logvar, z_q = model.forward(data)
        # 만약 마지막에 에러 -> z_q를 z['0'], z['1']로
        edges = logits

        """in DAG-GNN
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send)  # logits is of size: [num_sims, z_dims]
        edges = logits
        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, args.data_variable_size * args.x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)
        """
        
        # Forward VAE
        z = {}
        z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, output, x_mean, x_logvar, z_q = vae(data)
        # 만약 마지막에 에러 -> z_q를 z['0'], z['1']로

        if torch.sum(output != output):
            print('nan error \n')

        # ELBO: 어떻게?
        loss_nll = calculate_reconstruction_loss
        loss_kl = calculate_kl_loss

        loss = loss_nll + loss_kl

        # 여기서부턴 DAG-GNN의 추가 loss: 의미는 잘 모름. 일단 추가 -> 나중에 삭제
        # =======================
        # add A loss
        one_adj_A = origin_A # torch.mean(adj_A_tilt_decoder, dim =0)
        sparse_loss = args.tau_A * torch.sum(torch.abs(one_adj_A))

        # other loss term
        if args.use_A_connect_loss:
            connect_gap = A_connect_loss(one_adj_A, args.graph_threshold, z_gap)
            loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        if args.use_A_positiver_loss:
            positive_gap = A_positive_loss(one_adj_A, z_positive)
            loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

        # compute h(A)
        h_A = _h_A(origin_A, args.data_variable_size)
        loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A) + sparse_loss #+  0.01 * torch.sum(variance * variance)

        # DAG-GNN 참조: backward 및 추가 metrics, 마찬가지로 후에 수정

        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, args.tau_A*lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))

        mse_train.append(F.mse_loss(output, data).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        shd_train.append(shd)

    print(h_A.item())
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'shd_trian: {:.10f}'.format(np.mean(shd_train)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(vae.state_dict(), vae_indep_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'shd_trian: {:.10f}'.format(np.mean(shd_train)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    if 'graph' not in vars():
        print('error on assign')


    return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A

        



# 5. Save Model (checkpoint, etc.)


#===============
# MAIN
#===============

t_total = time.time()
best_ELBO_loss = np.inf
best_NLL_loss = np.inf
best_MSE_loss = np.inf
best_epoch = 0
best_ELBO_graph = []
best_NLL_graph = []
best_MSE_graph = []

if args.dependence_type == 0:
    c_A = args.c_A
    lambda_A = args.lambda_A
else:
    c_A = 0.0 # 훈련 모델 수정 필요 > c=0이면 optimizer update 오류?
    lambda_A = 0.0

h_A_new = torch.tensor(1.)
h_tol = args.h_tol
k_max_iter = int(args.k_max_iter)
h_A_old = np.inf