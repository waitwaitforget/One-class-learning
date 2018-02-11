import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FCEBM(nn.Module):
    '''
    Fully connected deep structured energy-based model
    '''
    def __init__(self, layershape, use_cuda=True):
        super(FCEBM, self).__init__()
        self.layershape = layershape
        self.use_cuda = use_cuda

        if use_cuda:
            self.global_bias = nn.Parameter(torch.rand(1, layershape[0]).cuda())
        else:
            self.global_bias = nn.Parameter(torch.rand(1, layershape[0]))
        
        self.z_dim = layershape[-1]
        def create_variable(dim1, dim2):
            if use_cuda:
                W = nn.Parameter(torch.rand(dim1, dim2).normal_(0, 0.01).cuda())
                b = nn.Parameter(torch.rand(dim2,).normal_(0, 0.01).cuda())
            else:
                W = nn.Parameter(torch.rand(dim1, dim2).normal_(0, 0.01))
                b = nn.Parameter(torch.rand(dim2,).normal_(0, 0.01))
            return W, b

        self.W, self.b = list(), list()

        for i in range(0, len(layershape)):
            W, b = create_variable(layershape[i-1], layershape[i])
            self.W.append(W)
            self.b.append(b)

        self.criterion = nn.MSELoss()

    def score(self, x):
        h = [None] * len(self.layershape)
        h[0] = x

        for i in range(1, len(self.layershape)):
            h[i] = F.softplus(torch.mm(h[i-1], self.W[i]) + self.b[i])
        
        encode = 0.5 * torch.pow(x - self.global_bias.expand_as(x), 2).sum(1) + h[-1].sum(1)
        return encode

    def forward(self, x):

        h = [None] * len(self.layershape)
        h[0] = x
        # forward propagation
        for i in range(1, len(self.layershape)):
            h[i] = F.softplus(torch.mm(h[i-1], self.W[i]) + self.b[i])
        # gradient of last layer
        result = Variable(torch.ones(x.size(0), self.z_dim))
        
        if self.use_cuda:
            result = result.cuda()
        
        for i in range(len(self.layershape)-2, -1, -1):
            result = torch.mm(result, self.W[i+1].transpose(1, 0)) * F.sigmoid(torch.mm(h[i+1], self.W[i+1].transpose(1,0)) + self.b[i])

        result = result + self.global_bias

        return torch.sum(torch.pow(result - x, 2), 1)