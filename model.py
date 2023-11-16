import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable
import torch.nn.utils.weight_norm as wn
import math
from utils import *
import numpy as np

from torch.autograd import Variable

class FlowLayer(nn.Module):
    """
    Flow layer class
    """
    def __init__(self, args):
        super(FlowLayer, self).__init__()
        self.args = args
    
    def forward(self, L, z):
        """
        :param L: batch_size (B) x latent_size^2 (L^2) from encoder output
        :param z: batch_size (B) x latent_size (L) from encoder output z0
        :return: z_new = L * z
        """
        # transform L to lower triangular matrix
        L_matrix = L.view(-1, self.args.z_size, self.args.z_size) # resize to get B x L x L
        LTmask = torch.tril(torch.ones(self.args.z_size, self.args.z_size), k=-1) # lower-triangular mask matrix
        I = Variable(torch.eye(self.args.z_size, self.args.z_size).expand(L_matrix.size(0), self.args.z_size, self.args.z_size))
        if self.args.cuda:
            LTmask = LTmask.cuda()
            I = I.cuda()
        LTmask = Variable(LTmask)
        LTmask = LTmask.unsqueeze(0).expand(L_matrix.size(0), self.args.z_size, self.args.z_size)
        LT = torch.mul(L_matrix, LTmask) + I # Lower triangular batches

        z_new = torch.bmm(LT, z.unsqueeze(2)).squeeze(2)

        return z_new

class Encoder(nn.Module):
    """
    Encoder class for VAE
    """
    def __init__(self, args, tol=0.1):
        super(Encoder, self).__init__()

        self.args = args
        self.adj_A = nn.Parameter(torch.from_numpy(args.adj_A).double(), requires_grad=True)
        self.Wa = nn.Parameter(torch.zeros(args.n_out), requires_grad=True)
        self.fc1 = nn.Linear(args.n_xdims, args.n_hid, bias=True)
        self.fc2 = nn.Linear(args.n_hid, args.n_out, bias=True)
        self.batch_size = args.batch_size

        self.z = nn.Parameter(torch.tensor(args.tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(args.adj_A)).double())

        # for other loss
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(args.adj_A)).double())


        # For Flow layer
        self.encoder_mean = nn.Linear(args.n_out, args.z_size)
        self.encoder_logvar = nn.Linear(args.n_out, args.z_size)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):

        if torch.sum(self.adj_A != self.adj_A):
            print('nan error \n')

        adj_A1 = torch.sinh(3.*self.adj_A) # amplify A

        # adj_Aforz = $I-A^T$
        adj_Aforz = (torch.eye(adj_A1.shape[0]).double() - (adj_A1.transpose(0,1)))


        adj_A = torch.eye(adj_A1.size()[0]).double()
        h0 = F.relu((self.fc1(inputs))) # first hidden layer
        h1 = (self.fc2(h0)) # second hidden layer
        logits = torch.matmul(adj_Aforz, h1 + self.Wa) - self.Wa

        # For Flow layer
        z_q_mean = self.encoder_mean(logits)
        z_q_logvar = self.encoder_logvar(logits)
        
        return z_q_mean, z_q_logvar, logits, adj_A1, adj_A, self.adj_A, self.z, self.z_positive, self.Wa




class Decoder(nn.Module):
    """
    Decoder class for VAE
    """

    def __init__(self, args):
        super(Decoder, self).__init__()

        self.out_fc1 = nn.Linear(args.z_size, args.n_hid, bias = True)
        self.out_fc2 = nn.Linear(args.n_hid, args.n_out, bias = True)

        self.batch_size = args.batch_size
        self.data_variable_size = args.data_variable_size

        self.decoder_mean = nn.Linear(args.n_out, args.data_variable_size)
        self.sigmoid = nn.Sigmoid()

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

        # to Mean
        x_mean = self.sigmoid(self.decoder_mean(out))
        x_logvar = 0.

        return mat_z, out, x_mean, x_logvar


class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()

        self.args = args
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.flow = FlowLayer(args)
        self.encoder_L = nn.Linear(args.n_out, args.z1_size**2)
        self.softmax = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() # standard deviation from log variance
        if self.args.cuda: # generate random noise epsilon for reparameterization trick
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()

        eps = Variable(eps)
        return eps.mul(std).add_(mu) # return y sample

    def forward(self, input):
        z = {}
        # z ~ q(z|x) : encoder

        z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa = self.encoder(input) 
        # myA = encoder.adj_A, adj_A_tilt is identity matrix

        # reparemeterization trick
        z['0'] = self.reparameterize(z_q_mean, z_q_logvar) # z0 : before Flow

        # Flow layer
        L = self.encoder_L(logits)
        z['1'] = self.flow(L, z['0']) # z1 : after Flow

        # z ~ p(x|z) : decoder
        mat_z, out, x_mean, x_logvar = self.decoder(z['1'], origin_A, Wa)

        return z_q_mean, z_q_logvar, logits, adj_A1, adj_A, adj_A_tilt, Wa, mat_z, out, x_mean, x_logvar, z['0'], z['1']