import time

import dhg

import dgl
import psutil
import torch
import random
import scipy.sparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
import torch.nn.functional as F
import os
from model import HGFS
from util.data import load_ACM_data
from util.tools import evaluate_results_nc
from utils import load_data
from util.pytorchtools import EarlyStopping

def score(logits, labels):  # 评分指标
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return accuracy, micro_f1, macro_f1  # micro f1和 macro f1是两个常用分类的评价指标


def evaluate(model, features, homo_dhg_graph, homo_graph, hp_G, labels, mask, loss_func, type_mask):#评价函数，调用上面的SCROE
    model.eval()
    with torch.no_grad():
        emb, logits, loss_unsupervised, _, _, _, _ = model(features, homo_dhg_graph,homo_graph, hp_G, type_mask)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])
    return loss, accuracy, micro_f1, macro_f1


def main(args):
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    # 只适用于同质图
    features, type_mask, labels, train_val_test_idx, homo_graph, homo_dhg_graph, meta_data_e, semantic_graph, hp_G \
        = load_ACM_data(device=args['device'])
    labels_plt = labels
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    feat_dim = [feat.shape[1] for feat in features]
    labels = torch.LongTensor(labels).to(args['device'])

    # 设定阈值
    threshold = 0.05
    # appnp的参数
    appnp_parameter = {'alpha': args['alpha'], 'edge_drop': args['edge_drop'], 'k_layers': args['k_layers']}

    # 开始训练 迭代好几次进行训练
    test_macro_f1_avg = 0
    test_micro_f1_avg = 0
    svm_macro_avg = np.zeros((7,), dtype=np.float)
    svm_micro_avg = np.zeros((7,), dtype=np.float)
    nmi_avg = 0
    ari_avg = 0

    for cur_repeat in range(args['repeat']):
        print('cur_repeat = {}   ==============================================================='.format(cur_repeat))
        model = HGFS(input_dim=feat_dim, hidden_dim=args['hidden_units'], num_classes=int(labels.max())+1,
                     dataset=args['dataset'], len_sub_graph=len(homo_dhg_graph), len_homo_graph=len(hp_G[0]),
                     len_relation_graph=len(hp_G[1]), threshold=threshold, appnp_parameter=appnp_parameter,
                     dropout=args['dropout'])
        model = model.to(args['device'])
        # from torchprof import Profile
        # early stop
        stopper = EarlyStopping(patience=args['patience'],
                                save_path='checkpoint/check_{}.pt'.format(args['dataset']))  # 通过耐心度来提前停止
        loss_fcn = torch.nn.CrossEntropyLoss()  # 损失函数为交叉熵损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])  # 通过adam梯度优化
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args['num_epochs'], max_lr=1e-3,
                                                        pct_start=0.05)
        train_step = 0
        for epoch in range(args['num_epochs']):
            emb, logits, loss_unsupervised, _, _, _, _ = model(features, homo_dhg_graph, homo_graph, hp_G, type_mask)
            # 为了应对过拟合 在损失函数中加入一个正则化项
            loss = loss_fcn(logits[train_idx], labels[train_idx])
            loss = loss + args["loss_alpha"] * loss_unsupervised
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()  # 更新所有参数
            # 计算内存消耗
            train_step += 1
            print(u'当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
            scheduler.step(train_step)

            val_loss, val_acc, val_micro_f1, val_macro_f1 = evaluate(model, features, homo_dhg_graph, homo_graph, hp_G, labels,
                                                                     val_idx, loss_fcn, type_mask)
            test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, features, homo_dhg_graph, homo_graph, hp_G, labels,
                                                                         test_idx, loss_fcn, type_mask)
            stopper(val_loss, model)
            # early_stop = stopper.step(val_loss.data.item(), val_acc, model)
            print('\tEpoch {:d} | Train Loss {:.4f}| Val Loss {:.4f} | Test Loss {:.4f}'.format(
                epoch + 1, loss.item(), val_loss.item(), test_loss.item()))
            if stopper.early_stop:
                print('Early stopping!')
                break
        model.load_state_dict(torch.load('checkpoint/check_{}.pt'.format(args['dataset'])))
        t_start = time.time()
        model.eval()
        with torch.no_grad():
            emb, logits, loss_unsupervised, relation_A, meta_path_A, att_meta, att_relation = model(features, homo_dhg_graph, homo_graph, hp_G, type_mask)
            test_loss, test_acc, test_micro_f1, test_macro_f1 = evaluate(model, features, homo_dhg_graph, homo_graph, hp_G, labels,
                                                                         test_idx, loss_fcn, type_mask)
            # relation_A = scipy.sparse.csr_matrix(relation_A)
            # meta_path_A = scipy.sparse.csr_matrix(meta_path_A)
            # scipy.sparse.save_npz("relation_acm", relation_A)
            # scipy.sparse.save_npz("meta_path_acm", meta_path_A)
            t_end = time.time()
            print('time:', (t_end-t_start))
            print("meta path att:", att_meta)
            print("relation att:", att_relation)
            print(test_macro_f1)
            print(test_micro_f1)
            test_macro_f1_avg = test_macro_f1_avg + test_macro_f1
            test_micro_f1_avg = test_micro_f1_avg + test_micro_f1

            svm_macro, svm_micro, nmi, ari = evaluate_results_nc(emb[test_idx].cpu().numpy(),
                                                                 labels[test_idx].cpu().numpy(), int(labels.max()) + 1)
            svm_macro_avg = svm_macro_avg + svm_macro
            svm_micro_avg = svm_micro_avg + svm_micro
            nmi_avg += nmi
            ari_avg += ari

    svm_macro_avg = svm_macro_avg / args['repeat']
    svm_micro_avg = svm_micro_avg / args['repeat']
    test_macro_f1_avg = test_macro_f1_avg / args['repeat']
    test_micro_f1_avg = test_micro_f1_avg / args['repeat']
    nmi_avg /= args['repeat']
    ari_avg /= args['repeat']
    print('---\nThe average of {} results:'.format(args['repeat']))
    print("macro-F1: ", test_macro_f1_avg)
    print("micro-F1: ", test_micro_f1_avg)
    print('Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]))
    print('Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]))
    print('NMI: {:.6f}'.format(nmi_avg))
    print('ARI: {:.6f}'.format(ari_avg))
    print('all finished')
    with open('log-ACM.txt', 'a+') as f:
        f.writelines('\n' + 'Macro-F1: ' + ', '.join(['{:.6f}'.format(macro_f1) for macro_f1 in svm_macro_avg]) + '\n' +
                     'Micro-F1: ' + ', '.join(['{:.6f}'.format(micro_f1) for micro_f1 in svm_micro_avg]) + '\n')
        f.writelines('NMI: ' + str(nmi_avg) + ',' + 'ARI: ' + str(ari_avg) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('HGFS')
    parser.add_argument('-ld', '--log-dir', type=str, default='results', help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true', help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--dataset', type=str, default='ACM_L', choices=['DBLP', 'DBLP_S', 'ACM_NEW', 'ACM', 'IMDB', 'ACM_L'])
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden_units', type=int, default=64, help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--loss_alpha', type=float, default=0.5, help='loss weight of unsupervised')
    parser.add_argument('--alpha', type=float, default=None, help='APPNP alpha')
    parser.add_argument('--edge_drop', type=float, default=0.05, help='APPNP edge drop')
    parser.add_argument('--k-layers', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--save-postfix', default='ACM', help='Postfix for the saved model and result. Default is ACM.')
    parser.add_argument('--repeat', type=int, default=1, help='重复训练和测试次数')
    args = parser.parse_args().__dict__
    # args = setup(args)
    print(args)
    main(args)
