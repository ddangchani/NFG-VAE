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
        LTmask = torch.tril(torch.ones(self.args.z_size, self.args.z_size), diagonal=-1) # lower-triangular mask matrix
        I = Variable(torch.eye(self.args.z_size, self.args.z_size).expand(L_matrix.size(0), self.args.z_size, self.args.z_size))
        if self.args.cuda:
            LTmask = LTmask.cuda()
            I = I.cuda()
        LTmask = Variable(LTmask)
        LTmask = LTmask.unsqueeze(0).expand(L_matrix.size(0), self.args.z_size, self.args.z_size)
        LT = torch.mul(L_matrix, LTmask) + I # Lower triangular batches
        z_new = torch.bmm(LT, z) # B x L x L * B x L x 1 = B x L x 1

        return z_new

class Encoder(nn.Module):
    """
    Encoder class for VAE
    """
    def __init__(self, args, adj_A, tol=0.1):
        super(Encoder, self).__init__()

        self.args = args
        self.adj_A = nn.Parameter(torch.from_numpy(adj_A).double(), requires_grad=True)
        self.Wa = nn.Parameter(torch.zeros(args.z_dims), requires_grad=True)
        self.fc1 = nn.Linear(args.x_dims, args.encoder_hidden, bias=True)
        self.fc2 = nn.Linear(args.encoder_hidden, args.z_dims, bias=True)
        self.batch_size = args.batch_size
        self.dropout = nn.Dropout(p=args.encoder_dropout)

        # for other loss
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())

        # For Flow layer
        self.encoder_mean = nn.Linear(args.z_dims, args.z_dims)
        self.encoder_logvar = nn.Linear(args.z_dims, args.z_dims)

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

        adj_A1 = torch.sinh(3.*self.adj_A.float()) # amplify A

        # adj_Aforz = $I-A^T$
        adj_Aforz = (torch.eye(adj_A1.shape[0]).float() - (adj_A1.transpose(0,1)))

        adj_A = torch.eye(adj_A1.size()[0]).float()
        h0 = F.relu((self.fc1(inputs.float()))) # first hidden layer
        h0 = self.dropout(h0)
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

        self.out_fc1 = nn.Linear(args.z_dims, args.decoder_hidden, bias = True)
        self.out_fc2 = nn.Linear(args.decoder_hidden, args.z_dims, bias = True)

        self.batch_size = args.batch_size
        self.node_size = args.node_size

        self.decoder_mean = nn.Linear(args.z_dims, args.z_dims)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.decoder_dropout)

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
        adj_A_new_inv = torch.inverse(torch.eye(origin_A.shape[0]).float() 
                                      - origin_A.transpose(0,1).float())

        mat_z = torch.matmul(adj_A_new_inv, input_z + Wa) - Wa
        
        H3 = F.relu((self.out_fc1(mat_z)))
        H3 = self.dropout(H3)
        out = self.out_fc2(H3)

        # to Mean
        x_mean = self.decoder_mean(out)
        x_logvar = 0.

        return mat_z, out, x_mean, x_logvar

class combination_L(nn.Module):
    def __init__(self,args):
        super(combination_L, self).__init__()
        self.args = args

    def forward(self, L, y):
        '''
        :param L: batch_size (B) x latent_size^2 * number_combination (L^2 * C)
        :param y: batch_size (B) x number_combination (C)
        :return: L_combination = y * L
        '''
        # calculate combination of Ls
        L_tensor = L.view(-1, self.args.z_size**2, self.args.number_combination ) # resize to get B x L^2 x C
        y = y.unsqueeze(1).expand(y.size(0), self.args.z_size**2, y.size(1)) # expand to get B x L^2 x C
        L_combination = torch.sum( L_tensor * y, 2 ).squeeze()
        return L_combination # B x L^2

class VAE_IAF(nn.Module):
    def __init__(self, args, adj_A):
        super(VAE_IAF, self).__init__()

        self.args = args
        self.encoder = Encoder(args, adj_A=adj_A)
        self.decoder = Decoder(args)
        self.flow = FlowLayer(args)
        self.encoder_L = nn.Linear(args.z_dims, args.z_size)
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

    def forward(self, input, rel_rec, rel_send):
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

        return z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, out, x_mean, x_logvar, z['0'], z['1'], L

class daggnn(nn.Module):
    def __init__(self, args, adj_A):
        super(daggnn, self).__init__()

        self.encoder = Encoder(args, adj_A=adj_A)
        self.decoder = Decoder(args)

        self.args = args
        self.adj_A = nn.Parameter(torch.from_numpy(adj_A).double(), requires_grad=True)
        self.Wa = nn.Parameter(torch.zeros(args.z_dims), requires_grad=True)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_() # standard deviation from log variance
        if self.args.cuda: # generate random noise epsilon for reparameterization trick
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()

        eps = Variable(eps)

        return eps.mul(std).add_(mu) # return y sample

    def forward(self, input, rel_rec, rel_send):
        z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa = self.encoder(input) 
        z = self.reparameterize(z_q_mean, z_q_logvar)
        mat_z, out, x_mean, x_logvar = self.decoder(z, origin_A, Wa)

        return z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, out, x_mean, x_logvar, z, z


class HF(nn.Module):
    def __init__(self):
        super(HF, self).__init__()

    def forward(self, v, z):
        '''
        :param v: batch_size (B) x latent_size (L)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = z - 2* v v_T / norm(v,2) * z
        '''
        # v * v_T
        vT = v.transpose(1,2) # v_T : transpose of v : B x 1 x L
        vvT = torch.bmm(v, vT) # vvT : batchdot( B x L x 1 * B x 1 x L ) = B x L x L
        
        # v * v_T * z
        vvTz = torch.bmm(vvT, z) # A * z : batchdot( B x L x L * B x L x 1 ).squeeze(2) = (B x L x 1).squeeze(2) = B x L
        # calculate norm ||v||^2
        norm_sq = torch.sum( v * v, 1 ) # calculate norm-2 for each row : B x 1
        norm_sq = norm_sq.expand(norm_sq.size(0), v.size(1)) # expand sizes : B x L

        # calculate new z
        z_new = z.squeeze(-1) - 2 * vvTz.squeeze(-1) / norm_sq # z - 2 * v * v_T  * z / norm2(v)
        return z_new.unsqueeze(-1)

class VAE_HF(nn.Module):
    def __init__(self, args, adj_A):
        super(VAE_HF, self).__init__()

        self.args = args
        self.encoder = Encoder(args, adj_A=adj_A)
        self.decoder = Decoder(args)
        self.softmax = nn.Softmax()
        self.HF = HF()
        # Householder flow
        self.v_layers = nn.ModuleList()
        # T > 0
        if self.args.number_of_flows > 0:
            for i in range(self.args.number_of_flows):
                self.v_layers.append(nn.Linear(self.args.z_size, self.args.z_size))

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

    def q_z_Flow(self, z, h_last):
        v = {}
        # Householder Flow:
        if self.args.number_of_flows > 0:
            v['1'] = self.v_layers[0](h_last.squeeze(-1)).unsqueeze(-1)
            z['1'] = self.HF(v['1'], z['0'])
            for i in range(1, self.args.number_of_flows):
                v[str(i + 1)] = self.v_layers[i](v[str(i)].squeeze(-1)).unsqueeze(-1)
                z[str(i + 1)] = self.HF(v[str(i + 1)], z[str(i)])
        
        return z

    def forward(self, input, rel_rec, rel_send):
        z = {}
        # z ~ q(z|x) : encoder
        z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa = self.encoder(input) 
        # myA = encoder.adj_A, adj_A_tilt is identity matrix

        # reparemeterization trick
        z['0'] = self.reparameterize(z_q_mean, z_q_logvar)

        # Flow layer
        z = self.q_z_Flow(z, logits)

        # z ~ p(x|z) : decoder
        mat_z, out, x_mean, x_logvar = self.decoder(z[str(self.args.number_of_flows)], origin_A, Wa)

        return z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, out, x_mean, x_logvar, z['0'], z[str(self.args.number_of_flows)]

class VAE_ccIAF(nn.Module):
    def __init__(self, args, adj_A):
        super(VAE_ccIAF, self).__init__()

        self.args = args
        self.encoder = Encoder(args, adj_A=adj_A)
        self.decoder = Decoder(args)
        self.flow = FlowLayer(args)
        self.encoder_L = nn.Linear(args.z_dims, args.z_size * args.number_combination)
        self.encoder_y = nn.Linear(args.z_size, args.number_combination)
        self.softmax = nn.Softmax()
        self.combination_L = combination_L(args)

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

    def forward(self, input, rel_rec, rel_send):
        z = {}
        # z ~ q(z|x) : encoder

        z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa = self.encoder(input) 
        # myA = encoder.adj_A, adj_A_tilt is identity matrix

        # reparemeterization trick
        z['0'] = self.reparameterize(z_q_mean, z_q_logvar) # z0 : before Flow

        # Flow layer
        L = self.encoder_L(logits)
        y = self.softmax(self.encoder_y(logits.squeeze(-1)))
        L_combination = self.combination_L(L, y)
        z['1'] = self.flow(L_combination, z['0']) # z1 : after Flow

        # z ~ p(x|z) : decoder
        mat_z, out, x_mean, x_logvar = self.decoder(z['1'], origin_A, Wa)

        return z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, out, x_mean, x_logvar, z['0'], z['1'], L_combination
