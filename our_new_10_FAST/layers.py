import math
from typing import Optional

import dgl
import torch
import torch as th
import torch_sparse
from dgl.nn.pytorch import EdgeWeightNorm
from torch import nn, Tensor
from dgl import function as fn
import torch.nn.functional as F
import numpy as np
from dgl.nn.pytorch import GraphConv, GATConv
from scipy.sparse import coo_matrix
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm, GCNConv
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptPairTensor, OptTensor, Adj
from torch_geometric.utils import spmm
from torch_sparse import SparseTensor


class APPNP(nn.Module):
    # 0.03 0.1 0.0
    # yelp
    def __init__(self, k_layers, alpha, edge_drop, dropout=0.6):
        super(APPNP, self).__init__()
        self.appnp = APPNPConv(k_layers, alpha, edge_drop)
        self.dropout = nn.Dropout(p=dropout)
        # self.dropout = dropout
        pass

    def forward(self, g, features, edge_weight=None):
        h = self.dropout(features)
        # h = F.dropout(features, self.dropout, training=self.training)
        h = self.appnp(g, h, edge_weight=edge_weight)
        return h


# 语义级别注意力
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention, self).__init__()
        # input:[Node, metapath, in_size]; output:[None, metapath, 1]; 所有节点在每个meta-path上的重要性值
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)    # 每个节点在metapath维度的均值; mean(0): 每个meta-path上的均值(/|V|); (MetaPath, 1)
        # beta = torch.tensor([[0.7], [0.2], [0.1]]).to(torch.device("cuda:0"))
        beta = torch.softmax(w, dim=0)       # 归一化          # (M, 1)
        # print(beta)
        beta = beta.expand((z.shape[0],) + beta.shape) #  拓展到N个节点上的metapath的值   (N, M, 1)
        return (beta * z).sum(1)     #  (beta * z)=>所有节点，在metapath上的attention值;    (beta * z).sum(1)=>节点最终的值      (N, D * K)


class SemanticAttention_1(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SemanticAttention_1, self).__init__()
        # input:[Node, metapath, in_size]; output:[None, metapath, 1]; 所有节点在每个meta-path上的重要性值
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)    # 每个节点在metapath维度的均值; mean(0): 每个meta-path上的均值(/|V|); (MetaPath, 1)
        # beta = torch.tensor([[0.7], [0.2], [0.1]]).to(torch.device("cuda:0"))
        beta = torch.softmax(w, dim=0)       # 归一化          # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) #  拓展到N个节点上的metapath的值   (N, M, 1)
        return (beta * z).sum(1)


class GAT(nn.Module):
    def __init__(self, in_dims, nhid, feat_drop, attn_drop, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.activation = F.elu
        self.gat_layers = GATConv(in_dims, nhid, nheads, feat_drop, attn_drop, activation=F.elu)

    def forward(self, g, h):
        # 调用GAT
        h, attention = self.gat_layers(g, h, get_attention=True)
        # h = self.gat_layers(g, h)
        return h


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.activation = F.elu
        self.gcn_layers = GraphConv(input_dim, output_dim, activation=self.activation)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, features, edge_weight=None):
        h = self.dropout(features)
        h = self.gcn_layers(graph, h, edge_weight=edge_weight)
        # h = self.dropout(h)
        # h = self.gcn_layers(graph, h, edge_weight=edge_weight)
        return h


# 带权重的APPNP
class APPNPConv(nn.Module):
    def __init__(self,
                 k,
                 alpha,
                 edge_drop=0.):
        super(APPNPConv, self).__init__()
        self._k = k
        self._alpha = alpha
        self.edge_drop = nn.Dropout(edge_drop)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if edge_weight is None:
                src_norm = th.pow(
                    graph.out_degrees().float().clamp(min=1), -0.5)
                shp = src_norm.shape + (1,) * (feat.dim() - 1)
                src_norm = th.reshape(src_norm, shp).to(feat.device)
                dst_norm = th.pow(
                    graph.in_degrees().float().clamp(min=1), -0.5)
                shp = dst_norm.shape + (1,) * (feat.dim() - 1)
                dst_norm = th.reshape(dst_norm, shp).to(feat.device)
            else:
                edge_weight = EdgeWeightNorm(
                    'both')(graph, edge_weight)
            feat_0 = feat
            for _ in range(self._k):
                # normalization by src node
                if edge_weight is None:
                    feat = feat * src_norm
                graph.ndata['h'] = feat
                w = th.ones(graph.number_of_edges(),
                            1) if edge_weight is None else edge_weight
                graph.edata['w'] = self.edge_drop(w).to(feat.device)
                graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                                 fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                # normalization by dst node
                if edge_weight is None:
                    feat = feat * dst_norm
                feat = (1 - self._alpha) * feat + self._alpha * feat_0
            return feat

# 得到节点之间的注意力
class AttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, cuda=False):
        super(AttentionLayer, self).__init__()
        self.dropout = dropout
        self.is_cuda = cuda

        self.W = nn.Parameter(nn.init.xavier_normal_(
            torch.Tensor(in_dim, hidden_dim).type(torch.cuda.FloatTensor if cuda else torch.FloatTensor),
            gain=np.sqrt(2.0)), requires_grad=True)
        self.W2 = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(hidden_dim, hidden_dim).type(
            torch.cuda.FloatTensor if cuda else torch.FloatTensor), gain=np.sqrt(2.0)),
            requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, bias, emb_dest, emb_src):
        # 目标节点和邻居节点的表示进行投影
        h_1 = torch.mm(emb_src, self.W)
        h_2 = torch.mm(emb_dest, self.W)
        # 算两个节点之间的注意力权重
        e = self.leakyrelu(torch.mm(torch.mm(h_2, self.W2), h_1.t()))
        zero_vec = -9e15 * torch.ones_like(e)
        # 两条边要有连接我才给你算注意力参数
        attention = torch.where(bias > 0, e, zero_vec)
        # 归一化之后的注意力参数
        attention = F.softmax(attention, dim=1)
        # 通过在训练集中使用dropout进一步计算注意力
        attention = F.dropout(attention, self.dropout, training=self.training)

        return attention


# 图级别注意力
class GraphChannelAttLayer(nn.Module):

    def __init__(self, num_channel, weights=None):
        super(GraphChannelAttLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_channel, 1, 1))
        # dblp imdb
        nn.init.constant_(self.weight, 0.1)  # equal weight
        # acm yelp mag
        # nn.init.uniform_(self.weight, a=-3, b=3)

        # if weights != None:
        #     self.weight.data = nn.Parameter(torch.Tensor(weights).reshape(self.weight.shape))
        #     with torch.no_grad():
        #         w = torch.Tensor(weights).reshape(self.weight.shape)
        #         self.weight.copy_(w)

    def forward(self, adj_list):
        adj_list = torch.stack(adj_list)
        # Row normalization of all graphs generated
        #
        adj_list = F.normalize(adj_list, dim=1, p=2)
        # print(F.softmax(self.weight, dim=0))
        # Hadamard product + summation -> Conv
        return torch.sum(adj_list * F.softmax(self.weight, dim=0), dim=0), F.softmax(self.weight, dim=0)


def coototensor(A):
    """
    Convert a coo_matrix to a torch sparse tensor
    """

    values = A.data
    indices = np.vstack((A.row, A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

#
def adj_matrix_weight_merge(A, adj_weight):
    """
    Multiplex Relation Aggregation
    """
    device = adj_weight[0].device
    adj_weight = F.softmax(adj_weight, dim=0)
    num = len(A)
    A_total = 0
    for i in range(num):
        tmp = A[i].multiply(adj_weight[i].item())
        A_total = A_total + tmp
    # A_total = A_total + A_total.T
    A_total_e = torch.tensor(A_total.data).type(torch.FloatTensor).to(device)
    A_total = dgl.from_scipy(A_total).to(device)

    return A_total, A_total_e


# 互信息鉴别器
class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        #features @ self.weight @ summary.t()
        return torch.matmul(features, torch.matmul(self.weight, summary))


# 适用于局部信息于局部信息互信息最大化的鉴别器
class Discriminator_local(nn.Module):
    def __init__(self, n_h):
        super(Discriminator_local, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, h1, h2):
        # 通过Bileaner计算h1和h2之间的局部互信息
        sc = self.f_k(h1, h2)
        return sc


# 稀疏dropout
class SparseDropout(nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - dprob

    def forward(self, x):
        mask = ((torch.rand(x._values().size()) + (self.kprob)).floor()).type(torch.bool)
        rc = x._indices()[:,mask]
        val = x._values()[mask]*(1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)


# 对比学习
class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, lam):
        super(Contrast, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, z_mp, z_sc, pos):
        # 两个视角下的表示进行投影
        z_proj_mp = self.proj(z_mp)
        z_proj_sc = self.proj(z_sc)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()

        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        lori_mp = -torch.log(matrix_mp2sc.mul(pos.to_dense()).sum(dim=-1)).mean()  # 各元素对应相乘

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos.to_dense()).sum(dim=-1)).mean()
        return self.lam * lori_mp + (1 - self.lam) * lori_sc


# 图卷积
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, alpha=None, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha1 = alpha
        self.alpha = nn.Parameter(torch.FloatTensor(2, 1, 1))
        torch.nn.init.uniform_(self.alpha, a=-3, b=3)
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.spmm(inputs, self.weight)
        output = torch.spmm(adj, support)
        # print(output.shape)
        # print(inputs.shape)
        # print(F.softmax(self.alpha, dim=0))
        if self.alpha1 == None:
            adj_list = torch.stack([output, inputs])
            output = torch.sum(adj_list * F.softmax(self.alpha, dim=0), dim=0)
        else:
            output = (1-self.alpha1)*output + self.alpha1*inputs
        if self.bias is not None:
            return F.elu(output + self.bias)
        else:
            return F.elu(output)


# 图注意力的稀疏矩阵形式
class FastGTConv(nn.Module):

    def __init__(self, in_channels, out_channels, args=None, pre_trained=None):
        super(FastGTConv, self).__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        self.bias = None
        # self.scale = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

        self.reset_parameters()

        if pre_trained is not None:
            with torch.no_grad():
                self.weight.data = pre_trained.weight.data

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.1)
        # nn.init.uniform_(self.weight, a=-0.6, b=0.6)
        # nn.init.constant_(self.weight, 0.3)
        # if self.args.non_local and self.args.non_local_weight != 0:
        #     with torch.no_grad():
        #         self.weight[:, -1] = self.args.non_local_weight
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # 前向传播
    def forward(self, A, num_nodes, epoch=None, layer=None):

        weight = self.weight
        filter = F.softmax(weight, dim=1)
        num_channels = filter.shape[0]
        results = []
        for i in range(num_channels):
            for j, (edge_index, edge_value) in enumerate(A):
                if j == 0:
                    total_edge_index = edge_index
                    total_edge_value = edge_value * filter[i][j]
                else:
                    total_edge_index = torch.cat((total_edge_index, edge_index), dim=1)
                    total_edge_value = torch.cat((total_edge_value, edge_value * filter[i][j]))

            index, value = torch_sparse.coalesce(total_edge_index.detach(), total_edge_value, m=num_nodes, n=num_nodes,
                                                 op='add')
            results.append((index, value))

        return results, filter



class AR_GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha):
        super(AR_GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.alpha1 = alpha
        self.alpha = nn.Parameter(torch.FloatTensor(2, 1, 1))
        torch.nn.init.uniform_(self.alpha, a=-3, b=3)

        # self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        output = self.conv1(x, edge_index, edge_weight)
        if self.alpha1 == None:
            adj_list = torch.stack([output, x])
            output = torch.sum(adj_list * F.softmax(self.alpha, dim=0), dim=0)
        else:
            output = (1-self.alpha1)*output + self.alpha1*x
        output = F.elu(output)
        return output
