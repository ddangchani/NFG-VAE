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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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
parser.add_argument('--graph_threshold', type= float, default = 0.3,  # 0.3 is good, 0.2 is error prune
                    help = 'threshold for learned adjacency matrix binarization')
parser.add_argument('--tau_A', type = float, default=0.0,
                    help='coefficient for L-1 norm of A.')
parser.add_argument('--lambda_A', type = float, default=0.,
                    help='coefficient for DAG constraint h(A).')
parser.add_argument('--c_A', type = float, default=1,
                    help='coefficient for absolute value h(A).')
parser.add_argument('--use_A_connect_loss',  type = int, default=0,
                    help='flag to use A connect loss')
parser.add_argument('--use_A_positiver_loss', type = int, default=0,
                    help = 'flag to enforce A must have positive values')
parser.add_argument('--cuda', type=int, default=0,
                    help='use cuda or not')
parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed.')
parser.add_argument('--epochs', type=int, default= 300,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default = 200, # note: should be divisible by sample size, otherwise throw an error
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=1e-3,  # basline rate = 1e-3
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--k_max_iter', type = int, default = 100,
                    help ='the max iteration number for searching lambda and c')
parser.add_argument('--encoder_dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder_dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default= 1.0,
                    help='LR decay factor.')
parser.add_argument('--h_tol', type=float, default = 1e-8,
                    help='the tolerance of error of h(A) to zero')
parser.add_argument('--x_dims', type=int, default=1, #changed here
                    help='The number of input dimensions: default 1.')
parser.add_argument('--z_dims', type=int, default=1,
                    help='The number of latent variable dimensions: default the same as variable size.')
parser.add_argument('--number_of_flows', type=int, default=5,
                    help='The number of HF flows: default 5.')
parser.add_argument('--flow_type', type=str, default='IAF',
                    help='The type of flows: "DAGGNN", "IAF", "HF"(Householder), "ccIAF"')
parser.add_argument('--lagrange', type=int, default=1,
                    help='Use lagrange multipliers or not.')
parser.add_argument('--number_combination', type=int, default=3,
                    help='The number of convex combinations: default 3.')
parser.add_argument('--loss_prevent', type=int, default=0,
                    help='Use loss that prevent overparametrization or not.')

args = parser.parse_args()
args.z_size = args.node_size # the number of latent variables
print(args)

# Device configuration (GPU or MPS or CPU)

if torch.cuda.is_available() and args.cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Seed
torch.manual_seed(args.seed)

# Folder to save the results, models, dataset

now = datetime.datetime.now().strftime('%m%d_%H%M')
if args.dependence_type == 1:
    folder = f'results/dependence/{now}_{args.flow_type}_node{args.node_size}_prop{int(args.dependence_prop*100)}_seed{args.seed}'
else:
    folder = f'results/independence/{args.graph_dist}/{now}_{args.flow_type}_node{args.node_size}_seed{args.seed}'

if not os.path.exists(folder):
    os.makedirs(folder)

# Save the arguments
meta_file = os.path.join(folder, 'meta.pkl')
model_file = os.path.join(folder, 'model.pt')

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

# save X to file
data_file = os.path.join(folder, 'data.pkl')
pickle.dump(X, open(data_file, 'wb'))

# save covariance matrix to file
if args.dependence_type == 1:
    cov_file = os.path.join(folder, 'cov.pkl')
    pickle.dump(cov, open(cov_file, 'wb'))

feat_train = torch.FloatTensor(X).to(device)
feat_valid = torch.FloatTensor(X).to(device)
feat_test = torch.FloatTensor(X).to(device)

train_data = TensorDataset(feat_train, feat_train)
valid_data = TensorDataset(feat_valid, feat_valid)
test_data = TensorDataset(feat_test, feat_train)

train_loader = DataLoader(train_data, batch_size=args.batch_size)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size)
test_loader = DataLoader(test_data, batch_size=args.batch_size)


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
if args.flow_type == 'IAF':
    vae = VAE_IAF(args=args, adj_A=adj_A)
elif args.flow_type == 'HF':
    vae = VAE_HF(args=args, adj_A=adj_A)
elif args.flow_type == 'DAGGNN':
    vae = daggnn(args=args, adj_A=adj_A)
elif args.flow_type == 'ccIAF':
    vae = VAE_ccIAF(args=args, adj_A=adj_A)
else:
    raise ValueError('Invalid flow type.')

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


def train(epoch, model, best_val_loss, G, lambda_A, c_A, optimizer, pbar=None):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_train = []
    
    # set model in training mode
    model.train()
    # scheduler.step()

    # update optimizer
    if args.lagrange:
        optimizer, lr = update_optimizer(optimizer, args.lr, c_A)

    z = {}

    # start training
    for batch_idx, (data, relations) in enumerate(train_loader):

        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data).double(), Variable(relations).double()

        # reshape data
        relations = relations.unsqueeze(2)

        optimizer.zero_grad()

        # Forward VAE
        z = {}
        if args.flow_type == 'IAF' or args.flow_type == 'ccIAF':
            z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, output, x_mean, x_logvar, z['0'], z['1'], LT = model(data, rel_rec, rel_send)
        else:
            z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, output, x_mean, x_logvar, z['0'], z['1'] = model(data, rel_rec, rel_send)

        """
        in DAG-GNN
        enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data, rel_rec, rel_send)  # logits is of size: [num_sims, z_dims]
        dec_x, output, adj_A_tilt_decoder = decoder(data, edges, args.data_variable_size * args.x_dims, rel_rec, rel_send, origin_A, adj_A_tilt_encoder, Wa)
        """

        if torch.sum(x_mean != x_mean):
            print('nan error \n')

        # KL Divergence Loss
        if args.loss_prevent == 0:
            loss_kl = calculate_kl_loss(z['0'], z['1'], z_q_mean, z_q_logvar, args.z_dims)
        else:
            loss_kl = calculate_kl_prevent(z['0'], z['1'], z_q_mean, z_q_logvar, args.z_dims)

        # Reconstruction Loss
        loss_nll = calculate_reconstruction_loss(x_mean, data, x_logvar)

        # add A loss
        sparse_loss = args.tau_A * torch.sum(torch.abs(origin_A))

        loss = loss_nll + loss_kl + sparse_loss
        
        # other loss term
        # if args.use_A_connect_loss:
        #     connect_gap = A_connect_loss(one_adj_A, args.graph_threshold, z_gap)
        #     loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

        # if args.use_A_positiver_loss:
        #     positive_gap = A_positive_loss(one_adj_A, z_positive)
        #     loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

        # compute h(A)
        if args.lagrange:
            h_A = _h_A(origin_A, args.node_size)
            loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A*origin_A)

        # DAG-GNN 참조: backward 및 추가 metrics, 마찬가지로 후에 수정

        loss.backward()
        loss = optimizer.step()

        origin_A.data = stau(origin_A.data, args.tau_A*args.lr)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().numpy()
        graph[np.abs(graph) < args.graph_threshold] = 0

        fdr, tpr, fpr, shd, nnz = count_accuracy(G, nx.DiGraph(graph))

        mse_train.append(F.mse_loss(x_mean, data).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        shd_train.append(shd)

    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []
    
    if pbar is None:
        print('Epoch: {:04d}'.format(epoch),
            'nll_train: {:.10f}'.format(np.mean(nll_train)),
            'kl_train: {:.10f}'.format(np.mean(kl_train)),
            'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
            'mse_train: {:.10f}'.format(np.mean(mse_train)),
            'shd_trian: {:.10f}'.format(np.mean(shd_train)),
            'time: {:.4f}s'.format(time.time() - t))
    
        if np.mean(nll_val) < best_val_loss:
            torch.save(model.state_dict(), model_file)
            print('Best model so far, saving...')
            print('Epoch: {:04d}'.format(epoch),
                'nll_train: {:.10f}'.format(np.mean(nll_train)),
                'kl_train: {:.10f}'.format(np.mean(kl_train)),
                'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
                'mse_train: {:.10f}'.format(np.mean(mse_train)),
                'shd_trian: {:.10f}'.format(np.mean(shd_train)),
                'time: {:.4f}s'.format(time.time() - t), file=log)
            log.flush()
    
    else:
        to_print = {'nll_train' : '{:.4f}'.format(np.mean(nll_train)),
                    'kl_train' : '{:.4f}'.format(np.mean(kl_train)),
                    'ELBO_loss' : '{:.4f}'.format(np.mean(kl_train)  + np.mean(nll_train)),
                    'shd_trian' : '{:.4f}'.format(np.mean(shd_train))}

        if args.lagrange:
            to_print['h_A'] = '{:.4f}'.format(h_A.item())
        
        pbar.set_description('Epoch: {:04d}'.format(epoch))
        pbar.set_postfix(to_print)
    
        if nll_val and (np.mean(nll_val) < best_val_loss):
            torch.save(model.state_dict(), model_file)
            pbar.write('Best model so far, saving...')
            pbar.write('Epoch: {:04d}'.format(epoch) +
                'nll_train: {:.10f}'.format(np.mean(nll_train)) +
                'kl_train: {:.10f}'.format(np.mean(kl_train)) +
                'ELBO_loss: {:.10f}'.format(np.mean(kl_train)  + np.mean(nll_train)) +
                'mse_train: {:.10f}'.format(np.mean(mse_train)) +
                'shd_trian: {:.10f}'.format(np.mean(shd_train)) +
                'time: {:.4f}s'.format(time.time() - t))
            log.flush()

        pbar.update(1)

    if 'graph' not in vars():
        print('error on assign')

    if args.flow_type == 'IAF':
        return np.mean(kl_train) + np.mean(nll_train), np.mean(nll_train), np.mean(mse_train), graph, origin_A, LT

    else:
        return np.mean(kl_train) + np.mean(nll_train), np.mean(nll_train), np.mean(mse_train), graph, origin_A, None


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

c_A = args.c_A 
lambda_A = args.lambda_A # 추후 검토

h_A_new = torch.tensor(1.)
h_tol = args.h_tol
k_max_iter = int(args.k_max_iter)
h_A_old = np.inf

pbar = tqdm(range(args.epochs * k_max_iter), desc='Training')

try:
    for step_k in range(k_max_iter):
        while c_A < 1e+20:
            for epoch in range(args.epochs):
                ELBO_loss, NLL_loss, MSE_loss, graph, origin_A, LT = train(epoch=epoch, model=vae, best_val_loss=best_ELBO_loss, G=G, lambda_A=lambda_A, c_A=c_A, optimizer=optimizer, pbar=pbar)
                if ELBO_loss < best_ELBO_loss:
                    best_ELBO_loss = ELBO_loss
                    best_epoch = epoch
                    best_ELBO_graph = graph

                if NLL_loss < best_NLL_loss:
                    best_NLL_loss = NLL_loss
                    best_epoch = epoch
                    best_NLL_graph = graph

                if MSE_loss < best_MSE_loss:
                    best_MSE_loss = MSE_loss
                    best_epoch = epoch
                    best_MSE_graph = graph

            # print("Optimization Finished!")
            # print("Best Epoch: {:04d}".format(best_epoch))
            if ELBO_loss > 2 * best_ELBO_loss:
                break

            # update parameters
            A_new = origin_A.data.clone()
            h_A_new = _h_A(A_new, args.node_size)
            if h_A_new.item() > 0.25 * h_A_old:
                c_A*=10
            else:
                break

        # update parameters
        # h_A, adj_A are computed in loss anyway, so no need to store
        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break

    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()



except KeyboardInterrupt:
    # print the best anway
    print(best_ELBO_graph)
    print(nx.to_numpy_array(G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(G, nx.DiGraph(best_ELBO_graph))
    print('Best ELBO Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz, file=log)

    print(best_NLL_graph)
    print(nx.to_numpy_array(G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(G, nx.DiGraph(best_NLL_graph))
    print('Best NLL Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz, file=log)

    print(best_MSE_graph)
    print(nx.to_numpy_array(G))
    fdr, tpr, fpr, shd, nnz = count_accuracy(G, nx.DiGraph(best_MSE_graph))
    print('Best MSE Graph Accuracy: fdr', fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz, file=log)

    graph = origin_A.data.clone().numpy()
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

# Save the Graph metrics
ls = ['ELBO', 'NLL', 'MSE']
for idx, graph_res in enumerate([best_ELBO_graph, best_NLL_graph, best_MSE_graph]):
    fdr, tpr, fpr, shd, nnz = count_accuracy(G, nx.DiGraph(graph_res))
    print('Best {} Graph Accuracy: fdr'.format(ls[idx]), fdr, ' tpr ', tpr, ' fpr ', fpr, 'shd', shd, 'nnz', nnz, file=log)

graph = origin_A.data.clone().numpy()
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
matG1 = np.matrix(origin_A.data.clone().numpy())
for line in matG1:
    np.savetxt(f1, line, fmt='%.5f')
f1.closed

# LT to pickle
pickle.dump(LT, open(folder + '/LT.pkl', 'wb'))

# Total training time
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total), file=log)

if log is not None:
    print(folder)
    log.close()