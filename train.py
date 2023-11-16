<<<<<<< Updated upstream
# What to do?
=======
import time
import datetime
import pickle
import os
import argparse

import torch
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset
import math
import numpy as np
from utils import *
from model import *
from data import *
>>>>>>> Stashed changes

from __future__ import print_function

from utils.distributions import log_Normal_diag, log_Normal_standard, log_Bernoulli

# 1. Arguments(argparse)



<<<<<<< Updated upstream
=======
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
>>>>>>> Stashed changes



# 2. Data Loading






# 3. Model Loading






# 4. Training Loop (epoch, batch, loss, optimizer, etc.)

<<<<<<< Updated upstream
=======
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
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
    model.train()

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

<<<<<<< Updated upstream
        # myA = encoder.adj_A, adj_A_tilt is identity matrix -> 왜 필요한가?
        z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, output, x_mean, x_logvar, z_q = model.forward(data)
        # 만약 마지막에 에러 -> z_q를 z['0'], z['1']로
        edges = logits

        """in DAG-GNN
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send)  # logits is of size: [num_sims, z_dims]
        edges = logits
        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, args.data_variable_size * args.x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)
        """
=======
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



        #===============================================

            
        
>>>>>>> Stashed changes
        
        if torch.sum(output != output):
            print('nan error\n')

        target = data
        preds = output
        variance = 0.

        # reconstruction accuracy loss - based on DAG-GNN
        loss_nll = nll_gaussian(preds, target, variance)

        # KL loss - 수정이 필요할지도
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss: 이것도 수정이 필요할지도
        loss = loss_kl + loss_nll

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
>>>>>>> Stashed changes


        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, args.tau_A*lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))


        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        shd_trian.append(shd)

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
          'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'shd_trian: {:.10f}'.format(np.mean(shd_trian)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()

    if 'graph' not in vars():
        print('error on assign')


    return np.mean(np.mean(kl_train)  + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A


# 5. Save Model (checkpoint, etc.)