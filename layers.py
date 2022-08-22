import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch_sparse
from torch_scatter import scatter_max, scatter_add

#--------------
### layers###
#--------------

class GraphConv(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        #for 3_D batch, need a loop!!!

        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphAttConv(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout):
        super(GraphAttConv, self).__init__()
        assert out_features % heads == 0
        out_perhead = out_features // heads

        self.graph_atts = nn.ModuleList([GraphAttConvOneHead(
               in_features, out_perhead, dropout=dropout) for _ in range(heads)])

        self.in_features = in_features
        self.out_perhead = out_perhead
        self.heads = heads

    def forward(self, input, adj):
        output = torch.cat([att(input, adj) for att in self.graph_atts], dim=1)
        # notice that original GAT use elu as activation func. 
        return output

    def __repr__(self):
        return self.__class__.__name__ + "({}->[{}x{}])".format(
                    self.in_features, self.heads, self.out_perhead)
   
class GraphAttConvOneHead(nn.Module):
    """
    Sparse version GAT layer, single head
    """

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttConvOneHead, self).__init__()
        self.weight = Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = Parameter(torch.zeros(size=(1, 2*out_features)))
        # init 
        nn.init.xavier_normal_(self.weight.data, gain=nn.init.calculate_gain('relu')) # look at here
        nn.init.xavier_normal_(self.a.data, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
         
    def forward(self, input, adj):
        edge = adj._indices()
        h = torch.mm(input, self.weight)
        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t() # edge_h: 2*D x E
        # do softmax for each row, this need index of each row, and for each row do softmax over it
        alpha = self.leakyrelu(self.a.mm(edge_h).squeeze()) # E
        n = len(input)
        alpha = self.softmax(alpha, edge[0], n)
        output = torch_sparse.spmm(edge, self.dropout(alpha), n, n, h) # h_prime: N x out
        # output = torch_sparse.spmm(edge, self.dropout(alpha), n, n, self.dropout(h)) # h_prime: N x out
        return output

    def softmax(self, src, index, num_nodes=None):
        """
        sparse softmax
        """
        num_nodes = index.max().item() + 1 if num_nodes is None else num_nodes
        out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
        out = out.exp()
        out = out / (scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
        return out

#--------------
### models ###
#--------------
class GNN_Encoder(nn.Module):
    def __init__(self, layer, nfeat, nhid, dropout, nhead=1, adj=None):
        super(GNN_Encoder, self).__init__()
        if layer == 'gcn':
            self.conv = GraphConv(nfeat, nhid)
            self.activation = nn.ReLU()
        elif layer == 'gat':
            self.conv = GraphAttConv(nfeat, nhid, nhead, dropout)
            self.activation = nn.ELU()
        
        self.dropout = nn.Dropout(p=dropout)
        self.adj = adj

    def forward(self, x, adj=None):
        if adj == None:
            adj = self.adj

        x = self.activation(self.conv(x, adj))
        output = self.dropout(x)

        return output

class GNN_Classifier(nn.Module):
    def __init__(self, layer, nhid, nclass, dropout, nhead=1, adj=None):
        super(GNN_Classifier, self).__init__()
        if layer == 'gcn':
            self.conv = GraphConv(nhid, nhid)
            self.activation = nn.ReLU()
        elif layer == 'gat':
            self.conv = GraphAttConv(nhid, nhid, nhead, dropout)
            self.activation = nn.ELU(True)
        
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
        self.adj = adj

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj=None, logit=False):
        if adj == None:
            adj = self.adj
        x = self.activation(self.conv(x, adj))
        x = self.dropout(x)
        if logit:
            return x
        x = self.mlp(x)
        
        return x

class MLP(nn.Module):
    def __init__(self, nhid, nclass):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(nhid, nclass)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x):
        x = self.mlp(x)

        return x

class Decoder(nn.Module):
    """
    Edge Reconstruction adopted in GraphSMOTE (https://arxiv.org/abs/2103.08826)
    """

    def __init__(self, nhid, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout
        self.de_weight = Parameter(torch.FloatTensor(nhid, nhid))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1,-2)))

        return adj_out