import this

import dgl
import scipy.sparse
import torch
from dgl.nn.pytorch import SGConv
import copy
from layers import *
import pandas as pd
from dgl.nn.pytorch.conv.tagconv import TAGConv
from sklearn.metrics.pairwise import cosine_similarity
from dhg.nn import HGNNPConv
from torch_geometric.nn import GCNConv
from dgl.nn.pytorch import GraphConv

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)


class HGFS(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_classes, dataset, len_sub_graph, len_homo_graph, len_relation_graph,
                 appnp_parameter, threshold, dropout=0.5):
        super(HGFS, self).__init__()
        self.dataset = dataset
        self.hidden_dim = hidden_dim
        self.len_sg = len_homo_graph
        self.dropout = dropout
        self.threshold = threshold
        alpha = appnp_parameter['alpha']

        # 线性层 将不同节点映射到相同空间
        self.fc_list = nn.ModuleList([nn.Linear(m, hidden_dim, bias=True) for m in input_dim])
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.hyper_gcn = nn.ModuleList()
        for i in range(len_sub_graph):
            self.hyper_gcn.append(HGNNPConv(hidden_dim, hidden_dim))

        self.hyper_gcn1 = HGNNPConv(hidden_dim, hidden_dim)

        self.hyper_gcn_homo = HGNNPConv(hidden_dim, hidden_dim)

        self.Semantic_Attention = SemanticAttention(hidden_dim, hidden_dim)

        self.concat = nn.Linear(2 * hidden_dim, hidden_dim)

        self.concat_hp = nn.Linear((len_sub_graph) *hidden_dim, hidden_dim, bias=True)
        self.concat_g = nn.Linear((len_homo_graph) * hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.activate = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        # acm 0.5 imdb: 0.15  dblp:都是0.01 acm:0.5
        # self.appnp_topo = APPNP(k_layers, alpha=alpha, edge_drop=edge_drop, dropout=dropout)
        # self.appnp_feat = APPNP(k_layers, alpha=alpha, edge_drop=edge_drop, dropout=dropout)
        # self.graph_attention_meta_path = GraphChannelAttLayer(len_homo_graph)
        # self.graph_attention_relation =  GraphChannelAttLayer(len_relation_graph)
        # self.concat = nn.Linear(2*hidden_dim, hidden_dim)
        self.drop_feat = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.gcn_meta_path = GraphConv(hidden_dim, hidden_dim)
        self.gcn_relation = GraphConv(hidden_dim, hidden_dim)
        # self.gcn_meta_path = GraphConvolution(hidden_dim, hidden_dim, alpha=alpha)
        # self.gcn_relation = GraphConvolution(hidden_dim, hidden_dim, alpha=alpha)
        # yelp 0.3 ACM 2.0 IMDB 1.5 DBLP: 0.9
        # IMDB 2.0 0.9
        self.contrast = Contrast(hidden_dim, 0.9, 0.5)

        self.fast_graph_att_sg = FastGTConv(len_homo_graph, 1)

        self.fast_graph_att_re = FastGTConv(len_relation_graph, 1)

    def forward(self, features_list, sub_graph, homo_graph, hp_G, type_mask):
        # 根据不同数据集区分目标节点
        if self.dataset == 'ACM_L':
            num = 4019
            num1 = 11246
        elif self.dataset == 'IMDB':
            num = 4278
            num1 = 11616
        elif self.dataset == 'DBLP':
            num = 4057
            num1 = 26128
        elif self.dataset == 'YELP':
            num = 2614
            num1 = 3913
        elif self.dataset == 'MAG':
            num = 4017
            num1 = 0
        elif self.dataset == 'FreeBase':
            num = 3492
            num1 = 0
        else:
            raise Exception("no such dataset!")
        device = features_list[0].device

        semantic_graph = hp_G[0]
        # semantic_graph, att_meta = self.graph_attention_meta_path(semantic_graph)

        semantic_graph, att_meta = self.fast_graph_att_sg(semantic_graph, num)
        # s_g = torch.zeros((num, num)).to(device)
        s_g = 0
        for i in range(len(semantic_graph)):
            a_edge, a_value = semantic_graph[i]
            mat_a = torch.sparse_coo_tensor(a_edge, a_value, (num, num)).to(a_edge.device)
            if i == 0:
                s_g = mat_a
            else:
                s_g += mat_a
        s_g = s_g.coalesce()
        s_g_weight = s_g.values()
        rows = s_g.indices()[0].detach().cpu().numpy()
        cols = s_g.indices()[0].detach().cpu().numpy()
        coo_mat = scipy.sparse.coo_matrix((s_g_weight.detach().cpu().numpy(), (rows, cols)),
                                shape=(num, num), dtype=np.float32)
        s_g = dgl.from_scipy(coo_mat).to(device)

        relation_graph = hp_G[1]
        relation_graph, att_relation = self.fast_graph_att_re(relation_graph, num1)
        r_g = 0
        for i in range(len(relation_graph)):
            a_edge, a_value = relation_graph[i]
            mat_a = torch.sparse_coo_tensor(a_edge, a_value, (num1, num1)).to(a_edge.device)
            if i == 0:
                r_g = mat_a
            else:
                r_g += mat_a
        r_g = r_g.coalesce()
        r_g_weight = r_g.values()
        rows = r_g.indices()[0].detach().cpu().numpy()
        cols = r_g.indices()[0].detach().cpu().numpy()
        coo_mat = scipy.sparse.coo_matrix((r_g_weight.detach().cpu().numpy(), (rows, cols)),
                                          shape=(num1, num1), dtype=np.float32)
        r_g = dgl.from_scipy(coo_mat).to(device)

        h = torch.zeros(type_mask.shape[0], self.hidden_dim, device=device)
        # 特征线性层
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            h[node_indices] = fc(features_list[i])

        h = self.activate(h)

        h = self.dropout(h)

        embeddings = []
        for i, g in enumerate(sub_graph):
            emb_t = self.hyper_gcn[i](h[0: g.num_v], g)[0:num]
            embeddings.append(emb_t.flatten(1))
        embeddings = torch.cat(embeddings, dim=1)
        embeddings = self.concat_hp(embeddings)
        embedding = self.activate(embeddings)
        embedding = self.dropout(embedding)
        embedding_all = torch.cat([embedding, h[num:]], dim=0)
        emb_relation = self.gcn_relation(r_g, embedding_all, edge_weight=r_g_weight)[0:num]
        emb_meta_path = self.gcn_meta_path(s_g, embedding, edge_weight=s_g_weight)
        emb = torch.stack([emb_relation, emb_meta_path], dim=1)
        emb = self.Semantic_Attention(emb)
        logits = self.classifier(emb)

        loss = self.contrast(emb_relation, emb_meta_path, homo_graph)

        return emb, logits, loss, relation_graph, semantic_graph, att_meta, att_relation
