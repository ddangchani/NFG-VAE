import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable
import torch.nn.utils.weight_norm as wn
import math


# What to do?

# args : arguments from train.py (parser)

# 1. VAE Encoder class

class Encoder(nn.Module):
    """
    Encoder class for VAE
    """
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(args.adj_A).double(), requires_grad=True))
        self.factor = args.factor

        self.Wa = nn.Parameter(torch.zeros(args.n_out), requires_grad=True) # Learnable parameter
        self.fc1 = nn.Linear(args.n_xdims, args.n_hid, bias = True)
        self.fc2 = nn.Linear(args.n_hid, args.n_out, bias = True)
        self.dropout_prob = args.do_prob
        self.batch_size = args.batch_size
        self.z = nn.Parameter(torch.tensor(args.tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(args.adj_A)).double())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, inputs, rel_rec, rel_send):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        # to amplify the value of A and accelerate convergence. (why?)
        adj_A1 = torch.sinh(3.*self.adj_A)

        # adj_Aforz = $I-A^T$
        adj_Aforz = (torch.eye(adj_A1.shape[0]).double() - (adj_A1.transpose(0,1)))

        adj_A = torch.eye(adj_A1.size()[0]).double()
        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x + self.Wa) - self.Wa

        return ...


# 2. Normalizing Flow layer class

class FlowLayer(nn.Module):
    """
    Normalizing Flow class (Inverse Autoregressive Flow)
    """

    def __init__(self, args):
        super(FlowLayer, self).__init__()
        n_in = args.f_size # f_size : flow size
        n_out = args.f_size * 2 + args.z_size * 2 # z_size : latent size

        self.z_size = args.z_size
        self.f_size = args.f_size
        self.args = args

        # self.down_ar_conv = 

    
    def up(self, input):
        
        return ...

    def down(self, input, sample=False):
            
        return ...


# 3. VAE Decoder class

class Decoder(nn.Module):
    """
    Decoder class for VAE
    """

    def __init__(self, args):
        super(Decoder, self).__init__()

        self.out_fc1 = nn.Linear(args.z_size, args.n_hid, bias = True)
        self.out_fc2 = nn.Linear(args.n_hid, args.n_out, bias = True)

        self.dropout_prob = args.do_prob
        self.batch_size = args.batch_size

        self.data_variable_size = args.data_variable_size

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_z, origin_A, Wa):
        # Wa : Encoder에서 학습한 parameter
        
        # adj_A_new_inv = $(I-A^T)^{-1}$
        adj_A_new_inv = torch.inverse(torch.eye(origin_A.shape[0]).double() 
                                      - origin_A.transpose(0,1))

        mat_z = torch.matmul(adj_A_new_inv, input_z + Wa) - Wa

        H3 = F.relu((self.out_fc1(mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out

        