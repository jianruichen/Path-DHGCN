import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, accuracy_score, precision_recall_curve,auc
import pickle
import random
from random import choice
from scipy import sparse

from torch import nn
def encode_onehot(labels):#encode
    classes = set(labels)#class label
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
#Load data and Preprocessing
def load_data(args,node):
    task=args.task
    data=args.data
    if task=="triangles":
        alltime = np.loadtxt("../datanew/" + args.data + "/" + args.data + "-times.txt")#时刻列表
        alltime = np.sort(alltime) - 1
        alltime = alltime.tolist()
        alltime = list(set(alltime))
        alltime = np.array(alltime)
        alltime = np.sort(alltime)
        alltime = alltime.tolist()
        train3 = np.loadtxt("../datanew/"+args.data+"/"+args.data+  "_3-train.txt")
        val3 = np.loadtxt("../datanew/"+args.data+"/"+args.data+ "_3-vali.txt")
        test3 = np.loadtxt("../datanew/"+args.data+"/"+args.data+ "_3-test.txt")
        train_list3,train_neg_list3=process_data3(args,node,train3)
        val_list3,val_neg_list3 = process_data3(args,node, val3)
        test_list3,test_neg_list3 = process_data3(args,node,  test3)
        return alltime, train_list3,train_neg_list3,val_list3,val_neg_list3,test_list3,test_neg_list3
    elif task == "quads":
        alltime = np.loadtxt("../datanew/" + args.data + "/" + args.data + "-times.txt")
        alltime = np.sort(alltime) - 1
        alltime = alltime.tolist()
        alltime = list(set(alltime))

        train4 = np.loadtxt("../datanew/"+args.data+"/"+args.data+ "_4-train.txt")
        val4 = np.loadtxt("../datanew/"+args.data+"/"+args.data+"_4-vali.txt")
        test4 = np.loadtxt("../datanew/"+args.data+"/"+args.data+"_4-test.txt")
        #处理数据
        #训练集
        train_list4,train_neg_list4=process_data4(args,node,train4)
        #验证集
        val_list4,val_neg_list4 = process_data4(args,node,  val4)
        #测试集
        test_list4,test_neg_list4 = process_data4(args,node, test4)
        return alltime, train_list4,train_neg_list4,val_list4,val_neg_list4,test_list4,test_neg_list4
    elif task == "pentagon":
        alltime = np.loadtxt("../datanew/" + args.data + "/" + args.data + "-times.txt")
        alltime = np.sort(alltime) - 1
        alltime = alltime.tolist()
        alltime = list(set(alltime))

        train5 = np.loadtxt("../datanew/" + args.data + "/" + args.data + "_5-train.txt")
        val5 = np.loadtxt("../datanew/" + args.data + "/" + args.data + "_5-vali.txt")
        test5 = np.loadtxt("../datanew/" + args.data + "/" + args.data + "_5-test.txt")
        # 处理数据
        # 训练集
        train_list5,train_neg_list5 = process_data5(args,node, train5)
        # 验证集
        val_list5,val_neg_list5 = process_data5(args,node, val5)
        # 测试集
        test_list5,test_neg_list5= process_data5(args,node,test5)
        return alltime,  train_list5,train_neg_list5,val_list5,val_neg_list5,test_list5,test_neg_list5
def norm(mx):#normalization
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx=sparse_mx_to_torch_sparse_tensor(mx)
    return mx
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
#3-order negative samples
def negative3(triads, times, node):
    node_list = np.arange(node)
    np.random.shuffle(node_list)
    triads = np.array(triads.tolist())
    neg_a1 = triads[:, 0]
    neg_a2 = np.random.choice(node_list, size=len(triads))
    neg_a3 = np.random.choice(node_list, size=len(triads))
    triads_time_neg = np.random.choice(times, size=len(triads))
    return list(zip(neg_a1, neg_a2, neg_a3, triads_time_neg))

#4-order negative samples
def negative4(quads, times, node):
    node_list = np.arange(node)
    np.random.shuffle(node_list)
    quads = np.array(quads.tolist())
    neg_a1 = quads[:, 0]
    neg_a2 = np.random.choice(node_list, size=len(quads))
    neg_a3 = np.random.choice(node_list, size=len(quads))
    neg_a4 = np.random.choice(node_list, size=len(quads))
    quads_time_neg = np.random.choice(times, size=len(quads))
    return list(zip(neg_a1, neg_a2, neg_a3, neg_a4, quads_time_neg))
#5-order negative samples
def negative5(pentogon, times, node):
    node_list = np.arange(node)
    np.random.shuffle(node_list)
    pentogon = np.array(pentogon.tolist())
    neg_a1 = pentogon[:, 0]
    neg_a2 = np.random.choice(node_list, size=len(pentogon))
    neg_a3 = np.random.choice(node_list, size=len(pentogon))
    neg_a4 = np.random.choice(node_list, size=len(pentogon))
    neg_a5 = np.random.choice(node_list, size=len(pentogon))
    pentogon_time_neg = np.random.choice(times, size=len(pentogon))
    return list(zip(neg_a1, neg_a2, neg_a3, neg_a4, neg_a5, pentogon_time_neg))

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
def accuracy(test_sign, prediction):#计算精度
    auc1 = roc_auc_score(test_sign, prediction)
    precision, recall, thresholds = precision_recall_curve(test_sign, prediction)
    AUC_PR = auc(recall, precision)
    return auc1, AUC_PR
#3-order feature fusion function
def features_fusion3(node_embedding,time_embedding,time,edges,neg_edges):
    ptime=list(edges[:, 3])
    pooles = [list(edges[:, i]) for i in range(4)]
    neg_pooles = [list(neg_edges[:, i]) for i in range(4)]
    time = [int(i) for i in time]
    time_dict = dict(zip(time, time_embedding))
    pooles[3] = [int(i) for i in pooles[3]]
    ptime_emb = [time_dict[ptime[i]] for i in range(len(pooles[3]))]
    ptime_emb = torch.tensor([item.cpu().detach().numpy() for item in ptime_emb])
    for i in range(3):
        pooles[i] = node_embedding[pooles[i]]
    for i in range(3):
        neg_pooles[i] = node_embedding[neg_pooles[i]]
    pos_all_emb = torch.cat(pooles[:3] + [ptime_emb], dim=1) / 4
    neg_all_emb = torch.cat(neg_pooles[:3] + [ptime_emb], dim=1) / 4
    all_emb = torch.cat([pos_all_emb, neg_all_emb], dim=0)
    labels_all = np.hstack([np.ones(len(pos_all_emb)), np.zeros(len(neg_all_emb))])
    labels_all = torch.tensor(labels_all)
    return all_emb, labels_all, ptime_emb
#4-order feature fusion function
def features_fusion4(node_embedding,time_embedding,time,edges,neg_edges):
    pos_indices = edges[:, :4].tolist()
    pos_times = edges[:, 4].tolist()
    neg_indices = neg_edges[:, :4].tolist()
    neg_times = neg_edges[:, 4].tolist()
    time = [int(t) for t in time]
    pos_times = [int(t) for t in pos_times]
    neg_times = [int(t) for t in neg_times]
    time_dict = dict(zip(time, time_embedding))
    pos_time_emb = [time_dict[ptime] for ptime in pos_times]
    pos_embs = [node_embedding[p_idx] for p_idx in pos_indices]
    pos_embs.append(torch.tensor(pos_time_emb).cpu().detach().numpy())
    pos_all_emb = torch.mean(torch.cat(pos_embs, dim=1), dim=1)
    neg_time_emb = [time_dict[ntime] for ntime in neg_times]
    neg_embs = [node_embedding[n_idx] for n_idx in neg_indices]
    neg_embs.append(torch.tensor(neg_time_emb).cpu().detach().numpy())
    neg_all_emb = torch.mean(torch.cat(neg_embs, dim=1), dim=1)
    all_emb = torch.cat([pos_all_emb, neg_all_emb], dim=0)
    labels_all = torch.tensor(np.hstack([np.ones(len(pos_all_emb)), np.zeros(len(neg_all_emb))]))
    return all_emb, labels_all, torch.tensor(pos_time_emb)
#5-order feature fusion function
def features_fusion5(node_embedding,time_embedding,time,edges,neg_edges):
    time = [int(i) for i in time]
    time_dict = dict(zip(time, time_embedding))
    # 整合所有节点的embedding
    node_embs = [node_embedding[np.array(edges[:, i].tolist())] for i in range(5)]
    ptime_emb = [time_dict[i] for i in map(int, edges[:, 5].tolist())]
    pos_all_emb = torch.cat(node_embs + [torch.tensor(ptime_emb)], dim=1) / 5
    # 同样整合所有neg节点的embedding
    neg_node_embs = [node_embedding[np.array(neg_edges[:, i].tolist())] for i in range(5)]
    neg_ptime_emb = [time_dict[i] for i in map(int, neg_edges[:, 5].tolist())]
    neg_all_emb = torch.cat(neg_node_embs + [torch.tensor(neg_ptime_emb)], dim=1) / 5
    all_emb = torch.cat([pos_all_emb, neg_all_emb], dim=0)
    labels_all = torch.tensor(np.hstack([np.ones(len(pos_all_emb)), np.zeros(len(neg_all_emb))]))
    return all_emb, labels_all, ptime_emb
#data processing
def process_data3(args,node,data):
    data = data - 1
    p3time = list(data[:, 3])
    edges = list(zip(list(data[:, 0]), list(data[:, 1]), list(data[:, 2])))
    edges = np.array(edges)
    neg_3 = negative3(edges, p3time, node)
    neg_3 = np.array(neg_3)
    return data,neg_3
def process_data4(args,node,data):
    data = data - 1
    p4time = list(data[:, 4])
    edges = list(zip(list(data[:, 0]), list(data[:, 1]), list(data[:, 2]), list(data[:, 3])))
    edges = np.array(edges)
    neg_4 = negative4(edges, p4time, node)
    neg_4 = np.array(neg_4)
    return data,neg_4
def process_data5(args,node,data):
    data=data-1
    p5time = list(data[:, 5])
    edges = list(zip(list(data[:, 0]), list(data[:, 1]), list(data[:, 2]), list(data[:, 3]),list(data[:, 4])))
    edges = np.array(edges)
    neg_5 = negative5(edges, p5time, node)
    neg_5 = np.array(neg_5)
    return data,neg_5
#Construction of Adjacency Matrix for Different Time Windows
def compare_pre_time_adj_train(window_num, node_number,data): 
    new_time_list = data[:, -1]
    time_unique = np.unique(new_time_list) #101个
    per_scale = np.round(len(time_unique)/window_num)
    time_length=int(len(data)/window_num)
    adj_list = []
    for i in range(window_num):
        up_raw = time_unique[int(per_scale * i)]
        if i != window_num-1:
            down_raw = time_unique[int(per_scale * (i + 1) - 1)]
            raw_num = np.where((new_time_list >= up_raw) & (new_time_list <= down_raw))
            edge_index_list = data[raw_num[0], 0:2]
        else:
            raw_num = np.where(new_time_list >= up_raw)
            edge_index_list = data[raw_num[0], 0:2]
        adj_list_single = sparse.csr_matrix(contruct_adj1(edge_index_list, node_number))
        adj_list.append(adj_list_single)
    return adj_list,time_length
#Time data processing
def newdata(args,node):
    train2 = np.loadtxt("../datanew/" + args.data + "/" + args.data + "_2-train.txt")
    adj_train_time,time_length_train = compare_pre_time_adj_train(args.windows, node, train2)
    return adj_train_time
