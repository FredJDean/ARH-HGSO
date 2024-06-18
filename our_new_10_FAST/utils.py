import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
import copy
import torch as th

from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio

import sys

sys.path.append('../../')
from data_loader import data_loader


def set_random_seed(seed=0):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """创建目录"""
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    """为目录名获取基于日期的后缀。"""
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)
    return post_fix


def setup_log_dir(args, sampling=False):
    """命名并创建日志目录。"""
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'
    mkdir_p(log_dir)
    return log_dir


# 根据论文所设置的参数
# default_configure = {
#     'lr': 0.001,  # Learning rate
#     'num_heads': [8],  # Number of attention heads for GAT
#     'hidden_units': 64,
#     'dropout': 0.6,
#     'weight_decay': 5e-4,
# }

sampling_configure = {'batch_size': 20}


# def setup(args):
#     args.update(default_configure)
#     # set_random_seed(args['seed'])
#     #args['log_dir'] = setup_log_dir(args)
#     return args
#
#
# def setup_for_sampling(args):
#     args.update(default_configure)
#     args.update(sampling_configure)
#     set_random_seed()
#     args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     args['log_dir'] = setup_log_dir(args, sampling=True)
#     return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_acm():
    dl = data_loader('C:/Users/Yanyeyu/Desktop/实验/数据处理/model/new_data/ACM')
    link_type_dic = {0: 'pa', 1: 'ap', 2: 'ps', 3: 'sp'}
    paper_num = dl.nodes['count'][0]    #目标节点数3025
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)



    features = th.FloatTensor(dl.nodes['attr'][0])
    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]+dl.labels_val['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)
    num_classes = 3
    #train_valid_mask = dl.labels_train['mask'][:paper_num]
    train_mask = dl.labels_train['mask'][:paper_num]
    valid_mask = dl.labels_val['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_indices = np.where(train_mask == True)[0]
    valid_indices = np.where(valid_mask == True)[0]
    test_indices = np.where(test_mask == True)[0]
    meta_paths = [['pa', 'ap'], ['ps', 'sp']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_acm_new():
    dl = data_loader('C:/Users/Yanyeyu/Desktop/实验/数据处理/model/new_data/ACM_NEW')
    link_type_dic = {0: 'pa', 1: 'ap', 2: 'ps', 3: 'sp'}
    paper_num = dl.nodes['count'][0]    #目标节点数3025
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    features = th.FloatTensor(dl.nodes['attr'][0])
    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]+dl.labels_val['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)
    num_classes = 3
    #train_valid_mask = dl.labels_train['mask'][:paper_num]
    train_mask = dl.labels_train['mask'][:paper_num]
    valid_mask = dl.labels_val['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_indices = np.where(train_mask == True)[0]
    valid_indices = np.where(valid_mask == True)[0]
    test_indices = np.where(test_mask == True)[0]
    meta_paths = [['pa', 'ap'], ['ps', 'sp']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths



def load_dblp(feat_type=0):
    prefix = 'C:/Users/Yanyeyu/Desktop/实验/数据处理/model/new_data/DBLP'
    dl = data_loader(prefix)
    link_type_dic = {0: 'pa', 1: 'ap', 2: 'pc', 3: 'cp'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    features = th.FloatTensor(dl.nodes['attr'][0])
    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]+dl.labels_val['data'][:author_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)
    num_classes = 4
    train_mask = dl.labels_train['mask'][:author_num]
    valid_mask = dl.labels_val['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_indices = np.where(train_mask == True)[0]
    valid_indices = np.where(valid_mask == True)[0]
    test_indices = np.where(test_mask == True)[0]
    meta_paths = [['ap', 'pa'],['ap', 'pc', 'cp', 'pa']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_dblp_s(feat_type=0):
    prefix = 'C:/Users/Yanyeyu/Desktop/实验/数据处理/model/new_data/DBLP_S'
    dl = data_loader(prefix)
    link_type_dic = {0: 'pa', 1: 'ap', 2: 'pc', 3: 'cp'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    features = th.FloatTensor(dl.nodes['attr'][0])
    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]+dl.labels_val['data'][:author_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)
    num_classes = 4
    train_mask = dl.labels_train['mask'][:author_num]
    valid_mask = dl.labels_val['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_indices = np.where(train_mask == True)[0]
    valid_indices = np.where(valid_mask == True)[0]
    test_indices = np.where(test_mask == True)[0]
    meta_paths = [['ap', 'pa'],['ap', 'pc', 'cp', 'pa']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths

def load_imdb():
    prefix = 'C:/Users/Yanyeyu/Desktop/实验/数据处理/model/new_data/IMDB'
    dl = data_loader(prefix)
    link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    features = th.FloatTensor(dl.nodes['attr'][0])
    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]+ dl.labels_val['data'][:movie_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)
    num_classes = 3
    train_mask = dl.labels_train['mask'][:movie_num]
    valid_mask = dl.labels_val['mask'][:movie_num]
    test_mask = dl.labels_test['mask'][:movie_num]
    train_indices = np.where(train_mask == True)[0]
    valid_indices = np.where(valid_mask == True)[0]
    test_indices = np.where(test_mask == True)[0]
    meta_paths = [['md', 'dm'], ['ma', 'am']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths

def load_yelp(feat_type=0):
    prefix = 'D:/STUDY/others/data/YELP_dat'
    dl = data_loader(prefix)
    link_type_dic = {0: 'bs', 1: 'sb', 2: 'bl', 3: 'lb', 4 : 'bu', 5:'ub'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    features = th.FloatTensor(dl.nodes['attr'][0])
    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]+dl.labels_val['data'][:author_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)
    num_classes = 3
    train_mask = dl.labels_train['mask'][:author_num]
    valid_mask = dl.labels_val['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_indices = np.where(train_mask == True)[0]
    valid_indices = np.where(valid_mask == True)[0]
    test_indices = np.where(test_mask == True)[0]
    meta_paths = [['bs', 'sb'], ['bl', 'lb'],['bu', 'ub']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_acm_l():
    dl = data_loader('D:/STUDY/others/data/ACM_L_dat')
    link_type_dic = {0: 'pa', 1: 'ap', 2: 'ps', 3: 'sp'}
    paper_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    features = th.FloatTensor(dl.nodes['attr'][0])
    labels = dl.labels_test['data'][:paper_num] + dl.labels_train['data'][:paper_num]+dl.labels_val['data'][:paper_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)
    num_classes = 3
    #train_valid_mask = dl.labels_train['mask'][:paper_num]
    train_mask = dl.labels_train['mask'][:paper_num]
    valid_mask = dl.labels_val['mask'][:paper_num]
    test_mask = dl.labels_test['mask'][:paper_num]
    train_indices = np.where(train_mask == True)[0]
    valid_indices = np.where(valid_mask == True)[0]
    test_indices = np.where(test_mask == True)[0]
    # pap psp
    meta_paths = [['pa', 'ap'], ['ps', 'sp']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_dblp_l():
    prefix = '../HGNN-AC-main/data/preprocessed/DBLP_L'
    dl = data_loader(prefix)
    link_type_dic = {0: 'ap', 1: 'pa', 2: 'pt', 3: 'tp',4:'pc',5:'cp'}
    author_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    features = th.FloatTensor(dl.nodes['attr'][0])
    labels = dl.labels_test['data'][:author_num] + dl.labels_train['data'][:author_num]+dl.labels_val['data'][:author_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)
    num_classes = 4
    train_mask = dl.labels_train['mask'][:author_num]
    valid_mask = dl.labels_val['mask'][:author_num]
    test_mask = dl.labels_test['mask'][:author_num]
    train_indices = np.where(train_mask == True)[0]
    valid_indices = np.where(valid_mask == True)[0]
    test_indices = np.where(test_mask == True)[0]
    meta_paths = [['ap', 'pa'], ['ap', 'pc', 'cp', 'pa'], ['ap', 'pt', 'tp', 'pa']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths



def load_imdb_l():
    prefix = 'D:/STUDY/others/data/IMDB_L_dat'
    dl = data_loader(prefix)
    link_type_dic = {0: 'md', 1: 'dm', 2: 'ma', 3: 'am'}
    movie_num = dl.nodes['count'][0]
    data_dic = {}
    for link_type in dl.links['data'].keys():
        src_type = str(dl.links['meta'][link_type][0])
        dst_type = str(dl.links['meta'][link_type][1])
        data_dic[(src_type, link_type_dic[link_type], dst_type)] = dl.links['data'][link_type].nonzero()
    hg = dgl.heterograph(data_dic)
    features = th.FloatTensor(dl.nodes['attr'][0])
    labels = dl.labels_test['data'][:movie_num] + dl.labels_train['data'][:movie_num]+ dl.labels_val['data'][:movie_num]
    labels = [np.argmax(l) for l in labels]
    labels = th.LongTensor(labels)
    num_classes = 3
    train_mask = dl.labels_train['mask'][:movie_num]
    valid_mask = dl.labels_val['mask'][:movie_num]
    test_mask = dl.labels_test['mask'][:movie_num]
    train_indices = np.where(train_mask == True)[0]
    valid_indices = np.where(valid_mask == True)[0]
    test_indices = np.where(test_mask == True)[0]
    meta_paths = [['md', 'dm'], ['ma', 'am']]
    return hg, features, labels, num_classes, train_indices, valid_indices, test_indices, \
           th.BoolTensor(train_mask), th.BoolTensor(valid_mask), th.BoolTensor(test_mask), meta_paths


def load_data(dataset):
    load_fun = None
    if dataset == 'ACM' :
        load_fun = load_acm
    elif dataset == 'ACM_NEW' :
        load_fun = load_acm_new
    elif dataset == 'DBLP' :
        load_fun = load_dblp
    elif dataset == 'DBLP_S' :
        load_fun = load_dblp_s
    elif dataset == 'IMDB':
        load_fun = load_imdb
    elif dataset == 'YELP':
        load_fun = load_yelp
    elif dataset=='ACM_L':
        load_fun = load_acm_l
    elif dataset=='DBLP_L':
        load_fun = load_dblp_l
    elif dataset=='IMDB_L':
        load_fun = load_imdb_l
    return load_fun()


class EarlyStopping(object):
    def __init__(self, patience=20):
        dt = datetime.datetime.now()
        self.filename = '.\\checkpoint\early_stop.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if loss <= self.best_loss:
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
            torch.save(model.state_dict(), 'HAN_acm_l.pt')
        return self.early_stop

    def save_checkpoint(self, model):
        """当val的loss下降时保存模型"""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """加载模型"""
        model.load_state_dict(torch.load(self.filename))
