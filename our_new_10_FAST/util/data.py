import dgl
import networkx as nx
import numpy as np
import scipy
import pickle
import torch
# import scipy
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse
import torch.nn.functional as F
from dhg import Graph, Hypergraph
from tqdm import tqdm


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# 构造超图
def load_ACM_data(device, prefix='D:/STUDY/others/data/ACM_processed'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()  # 节点类型0的特征，4019行4000列
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()  # 节点类型1的特征，7167行4000列
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()  # 节点类型2的特征，60行4000列

    features_list = [features_0, features_1, features_2]
    features_list = [torch.FloatTensor(feat).to(device) for feat in features_list]

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    labels = np.load(prefix + '/labels.npy')  # 加载标签，4019
    PA = scipy.sparse.load_npz(prefix + '/PA.npz')
    PS = scipy.sparse.load_npz(prefix + '/PS.npz')
    PP = scipy.sparse.load_npz(prefix + '/PP.npz')
    # PA = scipy.sparse.load_npz(prefix + '/attack_metapath/20%_1/pa.npz')
    # PS = scipy.sparse.load_npz(prefix + '/attack_metapath/20%_1/ps.npz')
    # PP = scipy.sparse.load_npz(prefix + '/attack_metapath/20%_1/pp.npz')
    HP_G = []
    PA_tmp = torch.from_numpy(np.vstack((PA.nonzero()[0], PA.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    PA_value = torch.FloatTensor(PA.data).to(device)

    PS_tmp = torch.from_numpy(np.vstack((PS.nonzero()[0], PS.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    PS_value = torch.FloatTensor(PS.data).to(device)

    PP_tmp = torch.from_numpy(np.vstack((PP.nonzero()[0], PP.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    PP_value = torch.FloatTensor(PP.data).to(device)

    HP_G.append((PA_tmp, PA_value))
    HP_G.append((PS_tmp, PS_value))
    HP_G.append((PP_tmp, PP_value))

    #
    # pp_g = dgl.DGLGraph(PP).to(device)
    # pa_g = dgl.DGLGraph(PA).to(device)
    # ps_g = dgl.DGLGraph(PS).to(device)
    #
    # HP_G = [PA, PS, PP]

    pa_edge_list = list(np.load(prefix + "/pa_edge_list.npy"))
    ps_edge_list = list(np.load(prefix + "/ps_edge_list.npy"))
    pp_edge_list = list(np.load(prefix + "/pp_edge_list.npy"))

    # print(pa_edge_list)
    G_pa = Graph(11246, pa_edge_list)
    G_ps = Graph(11246, ps_edge_list)
    G_pp = Graph(11246, pp_edge_list)

    # 将二部图转化为超图
    HP_G_pa = Hypergraph.from_graph_kHop(G_pa, k=1).to(device)
    HP_G_ps = Hypergraph.from_graph_kHop(G_ps, k=1).to(device)
    HP_G_pp = Hypergraph.from_graph_kHop(G_pp, k=1).to(device)

    pap = np.load(prefix + '/0/0-1-0.npy')
    psp = np.load(prefix + '/0/0-2-0.npy')
    # pap = scipy.sparse.load_npz(prefix + '/attack_metapath/20%_5/pap.npz').todense()
    # psp = scipy.sparse.load_npz(prefix + '/attack_metapath/20%_5/psp.npz').todense()

    pap = F.normalize(torch.from_numpy(pap).type(torch.FloatTensor))
    psp = F.normalize(torch.from_numpy(psp).type(torch.FloatTensor))

    pap = scipy.sparse.csr_matrix(pap)
    psp = scipy.sparse.csr_matrix(psp)

    pap_tmp = torch.from_numpy(np.vstack((pap.nonzero()[0], pap.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    pap_value = torch.FloatTensor(pap.data).to(device)

    psp_tmp = torch.from_numpy(np.vstack((psp.nonzero()[0], psp.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    psp_value = torch.FloatTensor(psp.data).to(device)

    semantic_graph = []
    semantic_graph.append((pap_tmp, pap_value))
    semantic_graph.append((psp_tmp, psp_value))

    homo_graph = pap + psp

    homo_graph[homo_graph > 0] = 1
    pos = scipy.sparse.csr_matrix(homo_graph)
    homo_graph = sparse_mx_to_torch_sparse_tensor(pos).to(device)

    # pap = scipy.sparse.csr_matrix(pap)
    # psp = scipy.sparse.csr_matrix(psp)
    # pap_e = torch.tensor(pap.data).type(torch.FloatTensor).to(device)
    # psp_e = torch.tensor(psp.data).type(torch.FloatTensor).to(device)
    # pap = dgl.DGLGraph(pap).to(device)
    # psp = dgl.DGLGraph(psp).to(device)

    pap_edge_list = list(np.load(prefix + "/pap_edge_list.npy"))
    psp_edge_list = list(np.load(prefix + "/psp_edge_list.npy"))
    G_pap = Graph(4019, pap_edge_list)
    G_psp = Graph(4019, psp_edge_list)

    HP_G_pap = Hypergraph.from_graph_kHop(G_pap, k=1).to(device)
    HP_G_psp = Hypergraph.from_graph_kHop(G_psp, k=1).to(device)

    # homo_graph = [pp_g, pa_g, ps_g, pap, psp]
    homo_dhg_graph = [HP_G_pp, HP_G_pa, HP_G_ps, HP_G_pap, HP_G_psp]
    # homo_graph_list =[pp_edge_list, pa_edge_list, ps_edge_list]
    # meta_data_e = [pap_e, psp_e]

    type_mask = np.load(prefix + '/node_types.npy')  # 行向量，11246，对应的00000,11111,222222

    HP_G = [semantic_graph, HP_G]
    meta_data_e = 0

    return features_list, type_mask, labels, train_val_test_idx, homo_graph, homo_dhg_graph, meta_data_e, semantic_graph, HP_G


def load_IMDB_data(device, prefix='D:/STUDY/others/data/IMDB_processed'):
    # 0 for movies, 1 for directors, 2 for actors
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()

    features_list = [features_0, features_1, features_2]
    features_list = [torch.FloatTensor(feat).to(device) for feat in features_list]

    MDM = np.load(prefix + '/0/0-1-0.npy')
    MAM = np.load(prefix + '/0/0-2-0.npy')

    # MDM = scipy.sparse.load_npz(prefix + '/attack_metapath/20%_5/mdm.npz').todense()
    # MAM = scipy.sparse.load_npz(prefix + '/attack_metapath/20%_5/mam.npz').todense()

    MDM = F.normalize(torch.from_numpy(MDM).type(torch.FloatTensor))
    MAM = F.normalize(torch.from_numpy(MAM).type(torch.FloatTensor))

    MDM = scipy.sparse.csr_matrix(MDM)
    MAM = scipy.sparse.csr_matrix(MAM)

    MDM_tmp = torch.from_numpy(np.vstack((MDM.nonzero()[0], MDM.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    MDM_value = torch.FloatTensor(MDM.data).to(device)
    MAM_tmp = torch.from_numpy(np.vstack((MAM.nonzero()[0], MAM.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    MAM_value = torch.FloatTensor(MAM.data).to(device)

    semantic_graph = []
    semantic_graph.append((MDM_tmp, MDM_value))
    semantic_graph.append((MAM_tmp, MAM_value))

    homo_graph = MDM + MAM
    homo_graph[homo_graph > 0] = 1
    homo_graph = scipy.sparse.csr_matrix(homo_graph)
    homo_graph = sparse_mx_to_torch_sparse_tensor(homo_graph).to(device)


    type_mask = np.load(prefix + '/node_types.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')  # 加载训练集，验证集，测试集的索引
    labels = np.load(prefix + '/labels.npy')  # 加载标签，4019
    MA = scipy.sparse.csr_matrix(np.load(prefix + "/MA_self_loop.npy"))
    MD = scipy.sparse.csr_matrix(np.load(prefix + "/MD_self_loop.npy"))

    # MA = scipy.sparse.load_npz(prefix + '/attack_metapath/20%_5/ma.npz')
    # MD = scipy.sparse.load_npz(prefix + '/attack_metapath/20%_5/md.npz')

    MA_tmp = torch.from_numpy(np.vstack((MA.nonzero()[0], MA.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    MA_value = torch.FloatTensor(MA.data).to(device)
    MD_tmp = torch.from_numpy(np.vstack((MD.nonzero()[0], MD.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    MD_value = torch.FloatTensor(MD.data).to(device)

    HP_G = []
    HP_G.append((MA_tmp, MA_value))
    HP_G.append((MD_tmp, MD_value))

    HP_G = [semantic_graph, HP_G]

    MA_edge_list = list(np.load(prefix + "/MA_edge_list.npy"))
    MD_edge_list = list(np.load(prefix + "/MD_edge_list.npy"))

    MAM_edge_list = list(np.load(prefix + "/mam_edge_list.npy"))
    MDM_edge_list = list(np.load(prefix + "/mdm_edge_list.npy"))

    G_MA = Graph(11616, MA_edge_list)
    G_MD = Graph(11616, MD_edge_list)

    G_MAM = Graph(4278, MAM_edge_list)
    G_MDM = Graph(4278, MDM_edge_list)

    HP_G_MA = Hypergraph.from_graph_kHop(G_MA, k=1).to(device)
    HP_G_MD = Hypergraph.from_graph_kHop(G_MD, k=1).to(device)

    HP_G_MAM = Hypergraph.from_graph_kHop(G_MAM, k=1).to(device)
    HP_G_MDM = Hypergraph.from_graph_kHop(G_MDM, k=1).to(device)

    homo_dhg_graph = [HP_G_MA, HP_G_MD, HP_G_MAM, HP_G_MDM]
    meta_data_e = 0

    return features_list, type_mask, labels, train_val_test_idx, homo_graph, homo_dhg_graph, meta_data_e, semantic_graph, HP_G


# 加载DBLP数据集
def load_DBLP_data(device, prefix='D:/STUDY/others/data/DBLP_processed'):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()  # （4057,334）
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()  # （14328,4231）
    features_2 = np.load(prefix + '/features_2.npy')  # (7723,50)
    features_3 = np.eye(20, dtype=np.float32)

    # adjM = scipy.sparse.load_npz(prefix + "/adjM.npz").toarray()

    features_list = [features_0, features_1, features_2, features_3]
    features_list = [torch.FloatTensor(feat).to(device) for feat in features_list]

    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')

    APVPA = np.load(prefix + '/0/A-P-V-P-A.npy')
    APA = np.load(prefix + '/0/A-P-A.npy')
    APTPA = np.load(prefix + '/0/A-P-T-P-A.npy')

    apvpa = APVPA
    apa = APA
    aptpa = APTPA

    APVPA = F.normalize(torch.from_numpy(APVPA).type(torch.FloatTensor))
    APA = F.normalize(torch.from_numpy(APA).type(torch.FloatTensor))
    APTPA = F.normalize(torch.from_numpy(APTPA).type(torch.FloatTensor))

    APVPA = scipy.sparse.csr_matrix(APVPA)
    APVPA_value = torch.FloatTensor(APVPA.data).to(device)

    APA = scipy.sparse.csr_matrix(APA)
    APA_value = torch.FloatTensor(APA.data).to(device)

    APTPA = scipy.sparse.csr_matrix(APTPA)
    APTPA_value = torch.FloatTensor(APTPA.data).to(device)


    semantic_graph = []
    APVPA_tmp = torch.from_numpy(np.vstack((APVPA.nonzero()[0], APVPA.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    semantic_graph.append((APVPA_tmp, APVPA_value))

    APA_tmp = torch.from_numpy(np.vstack((APA.nonzero()[0], APA.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    semantic_graph.append((APA_tmp, APA_value))

    APTPA_tmp = torch.from_numpy(np.vstack((APTPA.nonzero()[0], APTPA.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    semantic_graph.append((APTPA_tmp, APTPA_value))

    #
    HP_G = []
    AP = scipy.sparse.load_npz(prefix + "/AP.npz")
    AP_value = torch.FloatTensor(AP.data).to(device)
    AC = scipy.sparse.load_npz(prefix + "/AC.npz")
    AC_value = torch.FloatTensor(AC.data).to(device)
    AT = scipy.sparse.load_npz(prefix + "/AT.npz")
    AT_value = torch.FloatTensor(AT.data).to(device)

    AP_tmp = torch.from_numpy(np.vstack((AP.nonzero()[0], AP.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    AC_tmp = torch.from_numpy(np.vstack((AC.nonzero()[0], AC.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    AT_tmp = torch.from_numpy(np.vstack((AT.nonzero()[0], AT.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)

    HP_G.append((AP_tmp, AP_value))
    HP_G.append((AC_tmp, AC_value))
    HP_G.append((AT_tmp, AT_value))

    HP_G = [semantic_graph, HP_G]
    homo_graph = apa + apvpa + aptpa
    homo_graph[homo_graph > 0] = 1
    pos = scipy.sparse.csr_matrix(homo_graph)
    homo_graph = sparse_mx_to_torch_sparse_tensor(pos).to(device)
    # APVPA = scipy.sparse.csr_matrix(APVPA)
    # APA = scipy.sparse.csr_matrix(APA)
    # APTPA = scipy.sparse.csr_matrix(APTPA)
    # APVPA_e = torch.tensor(APVPA.data).type(torch.FloatTensor).to(device)
    # APA_e = torch.tensor(APA.data).type(torch.FloatTensor).to(device)
    # APTPA_e = torch.tensor(APTPA.data).type(torch.FloatTensor).to(device)
    # APVPA = dgl.DGLGraph(APVPA).to(device)
    # APA = dgl.DGLGraph(APA).to(device)
    # APTPA = dgl.DGLGraph(APTPA).to(device)
    # meta_data_e = [APVPA_e, APA_e, APTPA_e]

    #
    # AP = dgl.DGLGraph(AP).to(device)
    # AC = dgl.DGLGraph(AC).to(device)
    # # 本来就是个二部图
    # AT = dgl.DGLGraph(AT).to(device)

    AP_edge_list = list(np.load(prefix + "/AP_edge_list.npy"))
    AC_edge_list = list(np.load(prefix + "/AC_edge_list.npy"))
    AT_edge_list = list(np.load(prefix + "/AT_edge_list.npy"))

    APA_edge_list = list(np.load(prefix + "/APA_edge_list.npy"))
    APTPA_edge_list = list(np.load(prefix + "/APTPA_edge_list.npy"))
    APVPA_edge_list = list(np.load(prefix + "/APVPA_edge_list.npy"))

    G_ap = Graph(26128, AP_edge_list)
    G_ac = Graph(26128, AC_edge_list)
    G_at = Graph(26128, AT_edge_list)

    G_apa = Graph(4057, APA_edge_list)
    G_apcpa = Graph(4057, APVPA_edge_list)
    G_aptpa = Graph(4057, APTPA_edge_list)

    # 将二部图转化为超图 选的一阶邻居
    HP_G_ap = Hypergraph.from_graph_kHop(G_ap, k=1).to(device)
    HP_G_ac = Hypergraph.from_graph_kHop(G_ac, k=1).to(device)
    HP_G_at = Hypergraph.from_graph_kHop(G_at, k=1).to(device)

    HP_G_apa = Hypergraph.from_graph_kHop(G_apa, k=1).to(device)
    HP_G_apcpa = Hypergraph.from_graph_kHop(G_apcpa, k=1).to(device)
    HP_G_aptpa = Hypergraph.from_graph_kHop(G_aptpa, k=1).to(device)

    homo_dhg_graph = [HP_G_ac, HP_G_at, HP_G_ap, HP_G_apcpa, HP_G_aptpa, HP_G_apa]
    meta_data_e = 0
    return features_list, type_mask, labels, train_val_test_idx, homo_graph, homo_dhg_graph, meta_data_e, semantic_graph, HP_G


def load_YELP_data(device, prefix='D:/STUDY/others/data/YELP_processed'):
    # 0 for bussiness, 1 for users, 2 for services, 3 for rating levels
    # 保留B的特征 其他都用one-hot生成
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2.npz').toarray()
    features_3 = scipy.sparse.load_npz(prefix + '/features_3.npz').toarray()
    features_list = [features_0, features_1, features_2, features_3]
    features_list = [torch.FloatTensor(feat).to(device) for feat in features_list]

    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    # 我的
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npy').item()

    # Using PAP to define relations between papers.
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    semantic_graph = []
    BUB = np.load(prefix + "/0/b-u-b.npy")
    BSB = np.load(prefix + "/0/b-s-b.npy")
    BLB = np.load(prefix + "/0/b-l-b.npy")
    bub = BUB
    bsb = BSB
    blb = BLB
    BUB = F.normalize(torch.from_numpy(BUB).type(torch.FloatTensor))
    BSB = F.normalize(torch.from_numpy(BSB).type(torch.FloatTensor))
    BLB = F.normalize(torch.from_numpy(BLB).type(torch.FloatTensor))

    BUB = scipy.sparse.csr_matrix(BUB)
    BUB_value = torch.FloatTensor(BUB.data).to(device)
    BUB_tmp = torch.from_numpy(np.vstack((BUB.nonzero()[0], BUB.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    semantic_graph.append((BUB_tmp, BUB_value))

    BSB = scipy.sparse.csr_matrix(BSB)
    BSB_value = torch.FloatTensor(BSB.data).to(device)
    BSB_tmp = torch.from_numpy(np.vstack((BSB.nonzero()[0], BSB.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    semantic_graph.append((BSB_tmp, BSB_value))


    BLB = scipy.sparse.csr_matrix(BLB)
    BLB_value = torch.FloatTensor(BLB.data).to(device)
    BLB_tmp = torch.from_numpy(np.vstack((BLB.nonzero()[0], BLB.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    semantic_graph.append((BLB_tmp, BLB_value))

    # semantic_graph = [scipy.sparse.csr_matrix(s_g) for s_g in semantic_graph]
    BU = np.load(prefix + "/BU.npy")
    BS = np.load(prefix + "/BS.npy")
    BL = np.load(prefix + "/BL.npy")
    relation_A = []
    # BU = torch.FloatTensor(scipy.sparse.csr_matrix(BU).toarray())
    # BS = torch.FloatTensor(scipy.sparse.csr_matrix(BS).toarray())
    # BL = torch.FloatTensor(scipy.sparse.csr_matrix(BL).toarray())
    BU = scipy.sparse.csr_matrix(BU)
    BU_value = torch.FloatTensor(BU.data).to(device)
    BU_tmp = torch.from_numpy(np.vstack((BU.nonzero()[0], BU.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    relation_A.append((BU_tmp, BU_value))

    BS = scipy.sparse.csr_matrix(BS)
    BS_value = torch.FloatTensor(BS.data).to(device)
    BS_tmp = torch.from_numpy(np.vstack((BS.nonzero()[0], BS.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    relation_A.append((BS_tmp, BS_value))

    BL = scipy.sparse.csr_matrix(BL)
    BL_value = torch.FloatTensor(BL.data).to(device)
    BL_tmp = torch.from_numpy(np.vstack((BL.nonzero()[0], BL.nonzero()[1]))).type(torch.cuda.LongTensor).to(device)
    relation_A.append((BL_tmp, BL_value))
    # HP_G = [BU.to(device), BS.to(device), BL.to(device)]

    homo_graph = bub + bsb + blb
    homo_graph[homo_graph > 0] = 1
    # homo_graph[homo_graph <= 0.3] = 0
    # homo_graph = np.eye(2614)
    homo_graph = scipy.sparse.csr_matrix(homo_graph)
    homo_graph = sparse_mx_to_torch_sparse_tensor(homo_graph).to(device)

    HP_G = [semantic_graph, relation_A]

    BU_edge_list = list(np.load(prefix + "/BU_edge_list.npy"))
    BS_edge_list = list(np.load(prefix + "/BS_edge_list.npy"))
    BL_edge_list = list(np.load(prefix + "/BL_edge_list.npy"))

    BUB_edge_list = list(np.load(prefix + "/BUB_edge_list.npy"))
    BSB_edge_list = list(np.load(prefix + "/BSB_edge_list.npy"))
    BLB_edge_list = list(np.load(prefix + "/BLB_edge_list.npy"))

    G_bu = Graph(3913, BU_edge_list)
    G_bs = Graph(3913, BS_edge_list)
    G_bl = Graph(3913, BL_edge_list)

    G_bub = Graph(2614, BUB_edge_list)
    G_bsb = Graph(2614, BSB_edge_list)
    G_blb = Graph(2614, BLB_edge_list)

    # 将二部图转化为超图 选的一阶邻居
    HP_G_bu = Hypergraph.from_graph_kHop(G_bu, k=1).to(device)
    HP_G_bs = Hypergraph.from_graph_kHop(G_bs, k=1).to(device)
    HP_G_bl = Hypergraph.from_graph_kHop(G_bl, k=1).to(device)

    HP_G_bub = Hypergraph.from_graph_kHop(G_bub, k=1).to(device)
    HP_G_bsb = Hypergraph.from_graph_kHop(G_bsb, k=1).to(device)
    HP_G_blb = Hypergraph.from_graph_kHop(G_blb, k=1).to(device)

    homo_dhg_graph = [HP_G_bu, HP_G_bs, HP_G_bl, HP_G_bub, HP_G_bsb, HP_G_blb]
    meta_data_e = 0
    return features_list, type_mask, labels, train_val_test_idx, homo_graph, homo_dhg_graph, meta_data_e, semantic_graph, HP_G


def load_MAG_data(device, prefix='D:/STUDY/others/data/MAG'):
    features_p = np.load(prefix + '/features_p.npy')
    features_a = np.load(prefix + '/features_a.npy')
    features_i = np.load(prefix + '/features_i.npy')
    features_f = np.load(prefix + '/features_f.npy')

    features_list = [features_p, features_a, features_i, features_f]
    features_list = [torch.FloatTensor(feat).to(device) for feat in features_list]

    # 归一化之后的
    pap = np.load(prefix + '/adj_pap.npy')
    pfp = np.load(prefix + '/adj_pfp.npy')
    paiap = np.load(prefix + '/adj_paiap.npy')

    pap_homo = F.normalize(torch.from_numpy(pap).type(torch.FloatTensor))
    pfp_homo = F.normalize(torch.from_numpy(pfp).type(torch.FloatTensor))
    paiap_homo = F.normalize(torch.from_numpy(paiap).type(torch.FloatTensor))
    pap[pap<0.05] = 0
    pfp[pfp < 0.05] = 0
    paiap[paiap<0.05] = 0

    semantic_graph = [pap_homo.to(device), pfp_homo.to(device), paiap_homo.to(device)]

    homo_graph = pap + pfp + paiap
    homo_graph = scipy.sparse.csr_matrix(homo_graph)
    homo_graph = sparse_mx_to_torch_sparse_tensor(homo_graph).to(device)

    # adjM = scipy.sparse.load_npz(prefix + '/adjm.npz').toarray()

    #
    PA = scipy.sparse.load_npz(prefix + '/PA.npz')
    PF = scipy.sparse.load_npz(prefix + '/PF.npz')
    PI = scipy.sparse.load_npz(prefix + '/PI.npz')

    PA_homo = torch.FloatTensor(scipy.sparse.csr_matrix(PA).toarray())
    PF_homo = torch.FloatTensor(scipy.sparse.csr_matrix(PF).toarray())
    PI_homo = torch.FloatTensor(scipy.sparse.csr_matrix(PI).toarray())

    HP_G = [PA_homo.to(device), PF_homo.to(device), PI_homo.to(device)]

    # 标签
    labels = np.load(prefix + '/p_label.npy')
    train_idx = np.load(prefix + '/train_idx.npy')
    val_idx = np.load(prefix + '/val_idx.npy')
    test_idx = np.load(prefix + '/test_idx.npy')
    type_mask = np.zeros(26334)
    type_mask[4017: 19400] = 1
    type_mask[19400: 20880] = 2
    type_mask[20880: 26334] = 3
    type_mask = np.array(type_mask, dtype=np.int)

    # 构建超图
    row, col = PA.nonzero()
    PA_hyper = list(np.vstack((row, col)).T)

    row, col = PI.nonzero()
    PI_hyper = list(np.vstack((row, col)).T)

    row, col = PF.nonzero()
    PF_hyper = list(np.vstack((row, col)).T)

    G_PA = Graph(26334, PA_hyper)
    G_PI = Graph(26334, PI_hyper)
    G_PF = Graph(26334, PF_hyper)

    # 将二部图转化为超图 选的一阶邻居
    HP_G_PA = Hypergraph.from_graph_kHop(G_PA, k=1).to(device)
    HP_G_PI = Hypergraph.from_graph_kHop(G_PI, k=1).to(device)
    HP_G_PF = Hypergraph.from_graph_kHop(G_PF, k=1).to(device)

    # 将元路径转化为超图
    row, col = pap.nonzero()
    PAP_hyper = list(np.vstack((row, col)).T)

    row, col = pfp.nonzero()
    PFP_hyper = list(np.vstack((row, col)).T)

    row, col = paiap.nonzero()
    PAIAP_hyper = list(np.vstack((row, col)).T)

    G_PAP = Graph(4017, PAP_hyper)
    G_PAIAP = Graph(4017, PFP_hyper)
    G_PFP = Graph(4017, PAIAP_hyper)

    HP_G_PAP= Hypergraph.from_graph_kHop(G_PAP, k=1).to(device)
    HP_G_PAIAP = Hypergraph.from_graph_kHop(G_PAIAP, k=1).to(device)
    HP_G_PFP = Hypergraph.from_graph_kHop(G_PFP, k=1).to(device)

    homo_dhg_graph = [HP_G_PA, HP_G_PI, HP_G_PF, HP_G_PAP, HP_G_PAIAP, HP_G_PFP]
    HP_G = [semantic_graph, HP_G]

    return features_list, type_mask, labels, train_idx, val_idx, test_idx, homo_graph, homo_dhg_graph, semantic_graph, HP_G


def load_FreeBase_data(device,  prefix='D:/STUDY/others/data/FreeBase'):

    feat_M = np.eye(3492)
    feat_A = np.eye(33401)
    feat_D = np.eye(2502)
    feat_W = np.eye(4459)
    #
    features_list = [feat_M, feat_A, feat_D, feat_W]
    features_list = [torch.FloatTensor(feat).to(device) for feat in features_list]

    mam = scipy.sparse.load_npz(prefix + '/adj_mam.npz').toarray()
    mdm = scipy.sparse.load_npz(prefix + '/adj_mdm.npz').toarray()
    mwm = scipy.sparse.load_npz(prefix + '/adj_mwm.npz').toarray()

    homo_graph = mam + mdm + mwm
    homo_graph = scipy.sparse.csr_matrix(homo_graph)
    homo_graph = sparse_mx_to_torch_sparse_tensor(homo_graph).to(device)

    # 元路径下的子图
    mam_homo = F.normalize(torch.from_numpy(mam).type(torch.FloatTensor))
    mdm_homo = F.normalize(torch.from_numpy(mdm).type(torch.FloatTensor))
    mwm_homo = F.normalize(torch.from_numpy(mwm).type(torch.FloatTensor))

    #
    mam[mam < 0.05] = 0
    mdm[mdm < 0.05] = 0
    mwm[mwm < 0.05] = 0

    MA = scipy.sparse.load_npz(prefix + '/MA.npz')
    MD = scipy.sparse.load_npz(prefix + '/MD.npz')
    MW = scipy.sparse.load_npz(prefix + '/MW.npz')

    MA_homo = torch.FloatTensor(scipy.sparse.csr_matrix(MA).toarray())
    MD_homo = torch.FloatTensor(scipy.sparse.csr_matrix(MD).toarray())
    MW_homo = torch.FloatTensor(scipy.sparse.csr_matrix(MW).toarray())

    HP_G = [MA_homo.to(device), MD_homo.to(device), MW_homo.to(device)]

    labels = np.load(prefix + '/labels.npy')

    train_idx = np.array(range(600))

    val_idx = np.array(range(600, 900))

    test_idx = np.array(range(900, 3492))

    type_mask = np.load(prefix + '/node_types.npy')

    row, col = MA.nonzero()
    MA_hyper = list(np.vstack((row, col)).T)

    row, col = MD.nonzero()
    MD_hyper = list(np.vstack((row, col)).T)

    row, col = MW.nonzero()
    MW_hyper = list(np.vstack((row, col)).T)

    G_MA = Graph(43854, MA_hyper)
    G_MD = Graph(43854, MD_hyper)
    G_MW = Graph(43854, MW_hyper)

    # 将二部图转化为超图 选的一阶邻居
    HP_G_MA = Hypergraph.from_graph_kHop(G_MA, k=1).to(device)
    HP_G_MD = Hypergraph.from_graph_kHop(G_MD, k=1).to(device)
    HP_G_MW = Hypergraph.from_graph_kHop(G_MW, k=1).to(device)

    row, col = mam.nonzero()
    MAM_hyper = list(np.vstack((row, col)).T)

    row, col = mdm.nonzero()
    MDM_hyper = list(np.vstack((row, col)).T)

    row, col = mwm.nonzero()
    MWM_hyper = list(np.vstack((row, col)).T)

    G_MAM = Graph(3492, MAM_hyper)
    G_MDM = Graph(3492, MDM_hyper)
    G_MWM = Graph(3492, MWM_hyper)

    HP_G_MAM = Hypergraph.from_graph_kHop(G_MAM, k=1).to(device)
    HP_G_MDM = Hypergraph.from_graph_kHop(G_MDM, k=1).to(device)
    HP_G_MWM = Hypergraph.from_graph_kHop(G_MWM, k=1).to(device)

    homo_dhg_graph = [HP_G_MA, HP_G_MD, HP_G_MW, HP_G_MAM, HP_G_MDM, HP_G_MWM]
    semantic_graph = [mam_homo.to(device), mdm_homo.to(device), mwm_homo.to(device)]
    HP_G = [semantic_graph, HP_G]
    return features_list, type_mask, labels, train_idx, val_idx, test_idx, homo_graph, homo_dhg_graph, semantic_graph, HP_G