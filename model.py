import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.autograd import Variable
<<<<<<< Updated upstream
<<<<<<< Updated upstream
import torch.nn.utils.weight_norm as wn
import math
=======
from utils import *
>>>>>>> Stashed changes
=======
from utils import *
>>>>>>> Stashed changes


<<<<<<< Updated upstream
# What to do?
=======
# z로 변수변환 시 volume을 보존한다고 하는 convex combination (refer to Tomczak & Welling (2017))

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
        L_tensor = L.view( -1, self.args.z1_size**2, self.args.number_combination ) # resize to get B x L^2 x C
        y = y.unsqueeze(1).expand(y.size(0), self.args.z1_size**2, y.size(1)) # expand to get B x L^2 x C
        L_combination = torch.sum( L_tensor * y, 2 ).squeeze()
        return L_combination

# self.args.number_combination 을 설정해야 될지도

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
>>>>>>> Stashed changes

# args : arguments from train.py (parser)

# 1. VAE Encoder class

class Encoder(nn.Module):
    """
    Encoder class for VAE
    """
    def __init__(self, args, tol=0.1):
        super(Encoder, self).__init__()
        self.adj_A = nn.Parameter(Variable(torch.from_numpy(args.adj_A).double(), requires_grad=True))
        self.factor = args.factor

        self.Wa = nn.Parameter(torch.zeros(args.n_out), requires_grad=True) # Learnable parameter
        self.fc1 = nn.Linear(args.n_xdims, args.n_hid, bias = True)
        self.fc2 = nn.Linear(args.n_hid, args.n_out, bias = True)
        self.dropout_prob = args.do_prob
        self.batch_size = args.batch_size
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        self.z = nn.Parameter(torch.tensor(args.tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(args.adj_A)).double())
=======
        
        # for other loss term in training
=======

        # for other loss
>>>>>>> Stashed changes
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(args.adj_A)).double())

        # For Flow layer
        self.encoder_mean = nn.Linear(args.n_out, args.z1_size)
        self.encoder_logvar = nn.Linear(args.n_out, args.z1_size)


>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
        # adj_Aforz = $I-A^T$
        adj_Aforz = (torch.eye(adj_A1.shape[0]).double() - (adj_A1.transpose(0,1)))
=======
        adj_Aforz = preprocess_adj_new(adj_A1) # adj_Aforz = $I-A^T$
>>>>>>> Stashed changes

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
        
<<<<<<< Updated upstream
<<<<<<< Updated upstream
        return ...

    def down(self, input, sample=False):
            
        return ...
=======
        return z_q_mean, z_q_logvar, logits, adj_A1, adj_A, self.adj_A, self.z, self.z_positive, self.Wa
>>>>>>> Stashed changes
=======
        return z_q_mean, z_q_logvar, logits, adj_A1, adj_A, self.adj_A, self.z, self.z_positive, self.Wa
>>>>>>> Stashed changes


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
        
<<<<<<< Updated upstream
        # adj_A_new_inv = $(I-A^T)^{-1}$
        adj_A_new_inv = torch.inverse(torch.eye(origin_A.shape[0]).double() 
                                      - origin_A.transpose(0,1))
=======
        # adj_A_inv = $(I-A^T)^{-1}$
        adj_A_inv = preprocess_adj_new1(origin_A)
>>>>>>> Stashed changes

        mat_z = torch.matmul(adj_A_new_inv, input_z + Wa) - Wa

        H3 = F.relu((self.out_fc1(mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out

<<<<<<< Updated upstream
        
=======
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
<<<<<<< Updated upstream
        z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa = self.encoder(input) 
        # myA = encoder.adj_A, adj_A_tilt is identity matrix
=======
        z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa  = self.encoder(input)
>>>>>>> Stashed changes

        # reparemeterization trick
        z['0'] = self.reparameterize(z_q_mean, z_q_logvar) # z0 : before Flow

        # Flow layer
        # L_combination 추가함

        L = self.encoder_L(logits)
        y = self.softmax(self.encoder_y(logits))
        L_combination = self.combination_L(L, y)
        z['1'] = self.flow(L_combination, z['0']) # z1 : after Flow

        # z ~ p(x|z) : decoder
<<<<<<< Updated upstream
        mat_z, out, x_mean, x_logvar = self.decoder(z['1'], origin_A, Wa) 

        return z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, out, x_mean, x_logvar, z['0'], z['1']
>>>>>>> Stashed changes
=======
        mat_z, out, x_mean, x_logvar = self.decoder(z['1'], origin_A, Wa)

<<<<<<< Updated upstream
        return z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, out, x_mean, x_logvar, z['0'], z['1']
>>>>>>> Stashed changes
=======
        return z_q_mean, z_q_logvar, logits, origin_A, adj_A_tilt, myA, z_gap, z_positive, Wa, mat_z, out, x_mean, x_logvar, z['0'], z['1'], L_combination


## References/DAG-GNN/models.py


class MLPEncoder(nn.Module):
    """MLP encoder module."""
    def __init__(self, n_in, n_xdims, n_hid, n_out, adj_A, batch_size, do_prob=0., factor=True, tol = 0.1):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(Variable(torch.from_numpy(adj_A).double(), requires_grad=True))
        self.factor = factor

        self.Wa = nn.Parameter(torch.zeros(n_out), requires_grad=True)
        self.fc1 = nn.Linear(n_xdims, n_hid, bias = True)
        self.fc2 = nn.Linear(n_hid, n_out, bias = True)
        self.dropout_prob = do_prob
        self.batch_size = batch_size
        self.z = nn.Parameter(torch.tensor(tol))
        self.z_positive = nn.Parameter(torch.ones_like(torch.from_numpy(adj_A)).double())
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

        # to amplify the value of A and accelerate convergence.
        adj_A1 = torch.sinh(3.*self.adj_A)

        # adj_Aforz = I-A^T
        adj_Aforz = preprocess_adj_new(adj_A1)

        adj_A = torch.eye(adj_A1.size()[0]).double()
        H1 = F.relu((self.fc1(inputs)))
        x = (self.fc2(H1))
        logits = torch.matmul(adj_Aforz, x+self.Wa) -self.Wa

        print('logits', logits.shape)

        return x, logits, adj_A1, adj_A, self.z, self.z_positive, self.adj_A, self.Wa


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, n_in_z, n_out, encoder, data_variable_size, batch_size,  n_hid,
                 do_prob=0.):
        super(MLPDecoder, self).__init__()

        self.out_fc1 = nn.Linear(n_in_z, n_hid, bias = True)
        self.out_fc2 = nn.Linear(n_hid, n_out, bias = True)

        self.batch_size = batch_size
        self.data_variable_size = data_variable_size

        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, input_z, n_in_node, rel_rec, rel_send, origin_A, adj_A_tilt, Wa):

        #adj_A_new1 = (I-A^T)^(-1)
        adj_A_new1 = preprocess_adj_new1(origin_A)
        mat_z = torch.matmul(adj_A_new1, input_z+Wa)-Wa

        H3 = F.relu(self.out_fc1((mat_z)))
        out = self.out_fc2(H3)

        return mat_z, out, adj_A_tilt

>>>>>>> Stashed changes
