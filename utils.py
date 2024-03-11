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
def encode_onehot(labels):#编码
    classes = set(labels)#类别标签
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
#数据加载及处理
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
def norm(mx):#归一化
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
    return torch.sparse.FloatTensor(indices, values, shape)#将矩阵转换成向量
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
#三阶负样本
def negative3(triads,times,node):
    triads=triads.tolist()

    node_list = random.sample([i for i in range(0,node)],node)
    neg_a1 = []
    neg_a2 = []
    neg_a3 = []
    triads_time_neg=[]
    for i, j in enumerate(triads):
        b = triads[i]
        b[1] = choice(node_list)
        b[2] = choice(node_list)
        neg_a1.append(b[0])
        neg_a2.append(b[1])
        neg_a3.append(b[2])
        triads_time_neg.append(choice(times))
    return list(zip(neg_a1,neg_a2,neg_a3,triads_time_neg))
#四阶负样本
def negative4(quads,times,node):
    quads=quads.tolist()

    node_list = [random.randint(0, node-1) for i in range(node)]
    neg_a1 = []
    neg_a2 = []
    neg_a3 = []
    neg_a4 = []
    quads_time_neg=[]
    for i, j in enumerate(quads):
        b = quads[i]
        b[1] = choice(node_list)
        b[2] = choice(node_list)
        b[3] = choice(node_list)
        neg_a1.append(b[0])
        neg_a2.append(b[1])
        neg_a3.append(b[2])
        neg_a4.append(b[3])
        quads_time_neg.append(choice(times))
    return list(zip(neg_a1,neg_a2,neg_a3,neg_a4,quads_time_neg))
def negative5(pentogon,times,node):
    pentogon=pentogon.tolist()
    node_list = [random.randint(0, node-1) for i in range(node)]
    neg_a1 = []
    neg_a2 = []
    neg_a3 = []
    neg_a4 = []
    neg_a5 = []
    pentogon_time_neg=[]
    for i, j in enumerate(pentogon):
        b = pentogon[i]
        b[1] = choice(node_list)
        b[2] = choice(node_list)
        b[3] = choice(node_list)
        b[4] = choice(node_list)
        neg_a1.append(b[0])
        neg_a2.append(b[1])
        neg_a3.append(b[2])
        neg_a4.append(b[3])
        neg_a5.append(b[4])
        pentogon_time_neg.append(choice(times))
    return list(zip(neg_a1,neg_a2,neg_a3,neg_a4,neg_a5,pentogon_time_neg))
def normalize(mx):#归一化
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
#三阶特征融合函数
def features_fusion3(node_embedding,time_embedding,time,edges,neg_edges):
    p1=list(edges[:,0])
    p2 = list(edges[:, 1])
    p3 = list(edges[:, 2])
    ptime=list(edges[:, 3])
    neg_p1 = list(neg_edges[:, 0])
    neg_p2 = list(neg_edges[:, 1])
    neg_p3 = list(neg_edges[:, 2])
    neg_ptime=list(neg_edges[:, 3])
    time = [int(i) for i in time]
    time_dict=dict(zip(time, time_embedding))
    ptime_emb=[]
    ptime=[int(i) for i in ptime]
    for i, j in enumerate(ptime):
        ptime_emb.append(time_dict[ptime[i]])
    p1_emb=node_embedding[p1]
    p2_emb=node_embedding[p2]
    p3_emb=node_embedding[p3]
    ptime_emb=torch.tensor([item.cpu().detach().numpy() for item in ptime_emb])
    pos_all_emb = torch.cat([p1_emb, p2_emb], dim=1)
    pos_all_emb= torch.cat([pos_all_emb, p3_emb], dim=1)
    # pos_all_emb3 = torch.add(pos_all_emb2, p1_emb)
    pos_all_emb = torch.cat([pos_all_emb, ptime_emb], dim=1) / 3

    neg_p1_emb = node_embedding[neg_p1]
    neg_p2_emb = node_embedding[neg_p2]
    neg_p3_emb = node_embedding[neg_p3]
    neg_ptime_emb = []
    neg_ptime = [int(i) for i in neg_ptime]
    for i, j in enumerate(neg_ptime):
        neg_ptime_emb.append(time_dict[neg_ptime[i]])
    neg_ptime_emb = torch.tensor([item.cpu().detach().numpy() for item in neg_ptime_emb])
    neg_all_emb = torch.cat([neg_p1_emb, neg_p2_emb], dim=1)
    neg_all_emb= torch.cat([neg_all_emb, neg_p3_emb], dim=1)
    neg_all_emb = torch.cat([neg_all_emb, neg_ptime_emb], dim=1) / 3


    all_emb=torch.cat([pos_all_emb,neg_all_emb],dim=0)
    labels_all = np.hstack([np.ones(len(pos_all_emb)), np.zeros(len(neg_all_emb))])

    labels_all=torch.tensor(labels_all)
    return all_emb,labels_all,ptime_emb
def features_fusion4(node_embedding,time_embedding,time,edges,neg_edges):
    p1 = list(edges[:, 0])
    p2 = list(edges[:, 1])
    p3 = list(edges[:, 2])
    p4 = list(edges[:, 3])
    ptime = list(edges[:, 4])
    neg_p1 = list(neg_edges[:, 0])
    neg_p2 = list(neg_edges[:, 1])
    neg_p3 = list(neg_edges[:, 2])
    neg_p4 = list(neg_edges[:, 3])
    neg_ptime = list(neg_edges[:, 4])
    time = [int(i) for i in time]
    time_dict=dict(zip(time, time_embedding))
    ptime_emb=[]
    ptime=[int(i) for i in ptime]
    for i, j in enumerate(ptime):
        ptime_emb.append(time_dict[ptime[i]])
    p1_emb=node_embedding[p1]
    p2_emb=node_embedding[p2]
    p3_emb=node_embedding[p3]
    p4_emb = node_embedding[p4]
    ptime_emb=torch.tensor([item.cpu().detach().numpy() for item in ptime_emb])

    pos_all_emb1=torch.cat([p1_emb,p2_emb],dim=1)
    pos_all_emb2=torch.cat([pos_all_emb1,p3_emb],dim=1)
    pos_all_emb3 = torch.cat([pos_all_emb2,p4_emb],dim=1)
    pos_all_emb = torch.cat([pos_all_emb3,ptime_emb],dim=1)/4
    neg_p1_emb = node_embedding[neg_p1]
    neg_p2_emb = node_embedding[neg_p2]
    neg_p3_emb = node_embedding[neg_p3]
    neg_p4_emb = node_embedding[neg_p4]
    neg_ptime_emb = []
    neg_ptime = [int(i) for i in neg_ptime]
    for i, j in enumerate(neg_ptime):
        neg_ptime_emb.append(time_dict[neg_ptime[i]])
    neg_ptime_emb = torch.tensor([item.cpu().detach().numpy() for item in neg_ptime_emb])
    neg_all_emb1 = torch.cat([neg_p1_emb, neg_p2_emb],dim=1)
    neg_all_emb2 = torch.cat([neg_all_emb1, neg_p3_emb],dim=1)
    neg_all_emb3 = torch.cat([neg_all_emb2, neg_p4_emb],dim=1)
    neg_all_emb = torch.cat([neg_all_emb3, neg_ptime_emb],dim=1)/4
    all_emb=torch.cat([pos_all_emb,neg_all_emb],dim=0)
    labels_all = np.hstack([np.ones(len(pos_all_emb)), np.zeros(len(neg_all_emb))])
    labels_all=torch.tensor(labels_all)
    return all_emb,labels_all,ptime_emb
def features_fusion5(node_embedding,time_embedding,time,edges,neg_edges):
    p1 = list(edges[:, 0])
    p2 = list(edges[:, 1])
    p3 = list(edges[:, 2])
    p4 = list(edges[:, 3])
    p5 = list(edges[:, 4])
    ptime = list(edges[:, 5])
    neg_p1 = list(neg_edges[:, 0])
    neg_p2 = list(neg_edges[:, 1])
    neg_p3 = list(neg_edges[:, 2])
    neg_p4 = list(neg_edges[:, 3])
    neg_p5 = list(neg_edges[:, 4])
    neg_ptime = list(neg_edges[:, 5])
    time = [int(i) for i in time]
    time_dict=dict(zip(time, time_embedding))
    ptime_emb=[]
    ptime=[int(i) for i in ptime]
    for i, j in enumerate(ptime):
        ptime_emb.append(time_dict[ptime[i]])
    p1_emb=node_embedding[p1]
    p2_emb=node_embedding[p2]
    p3_emb=node_embedding[p3]
    p4_emb = node_embedding[p4]
    p5_emb = node_embedding[p5]
    ptime_emb=torch.tensor([item.cpu().detach().numpy() for item in ptime_emb])
    pos_all_emb1=torch.cat([p1_emb,p2_emb],dim=1)
    pos_all_emb2=torch.cat([pos_all_emb1,p3_emb],dim=1)
    pos_all_emb3 = torch.cat([pos_all_emb2,p4_emb],dim=1)
    pos_all_emb4 = torch.cat([pos_all_emb3, p5_emb],dim=1)
    pos_all_emb = torch.cat([pos_all_emb4,ptime_emb],dim=1)/5

    neg_p1_emb = node_embedding[neg_p1]
    neg_p2_emb = node_embedding[neg_p2]
    neg_p3_emb = node_embedding[neg_p3]
    neg_p4_emb = node_embedding[neg_p4]
    neg_p5_emb = node_embedding[neg_p5]
    neg_ptime_emb = []
    neg_ptime = [int(i) for i in neg_ptime]
    for i, j in enumerate(neg_ptime):
        neg_ptime_emb.append(time_dict[neg_ptime[i]])
    neg_ptime_emb = torch.tensor([item.cpu().detach().numpy() for item in neg_ptime_emb])
    neg_all_emb1 = torch.cat([neg_p1_emb, neg_p2_emb],dim=1)
    neg_all_emb2 = torch.cat([neg_all_emb1, neg_p3_emb],dim=1)
    neg_all_emb3 = torch.cat([neg_all_emb2, neg_p4_emb],dim=1)
    neg_all_emb4 = torch.cat([neg_all_emb3, neg_p5_emb],dim=1)
    neg_all_emb = torch.cat([neg_all_emb4, neg_ptime_emb],dim=1)/5
    all_emb=torch.cat([pos_all_emb,neg_all_emb],dim=0)
    labels_all = np.hstack([np.ones(len(pos_all_emb)), np.zeros(len(neg_all_emb))])
    labels_all=torch.tensor(labels_all)
    return all_emb,labels_all,ptime_emb
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
    #五阶
    p5time = list(data[:, 5])
    edges = list(zip(list(data[:, 0]), list(data[:, 1]), list(data[:, 2]), list(data[:, 3]),list(data[:, 4])))
    edges = np.array(edges)
    neg_5 = negative5(edges, p5time, node)
    neg_5 = np.array(neg_5)
    return data,neg_5
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)#将矩阵转换成向量
def contruct_adj1(edges, node_num):
    adj_time = np.zeros(shape=(node_num, node_num))
    for i in range(edges.shape[0]):
        # adj_time[int(edges[i, 0]-1), int(edges[i, 1]-1)] = adj_time[int(edges[i, 0]-1), int(edges[i, 1]-1)] + 1
        adj_time[int(edges[i, 0] - 1), int(edges[i, 1] - 1)] = 1
        adj_time[int(edges[i, 1] - 1), int(edges[i, 0] - 1)] = 1
    return adj_time
def compare_pre_time_adj_train(window_num, node_number,data):  #data是三列信息，节点，节点，时间
    new_time_list = data[:, -1]
    time_unique = np.unique(new_time_list) #101个
    per_scale = np.round(len(time_unique)/window_num)#每个窗口的时间间隔
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
def compare_pre_time_adj_val(node_number,data,time_length):  #data是三列信息，节点，节点，时间
    new_time_list = data[:, -1]
    time_unique = np.unique(new_time_list) #101个
    window_num=int(len(data)/time_length)
    per_scale = np.round(len(time_unique)/window_num)#每个窗口的时间间隔
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
    return adj_list
def compare_pre_time_adj_test(node_number,data,time_length):  #data是三列信息，节点，节点，时间
    new_time_list = data[:, -1]
    time_unique = np.unique(new_time_list) #101个
    window_num = int(len(data) / time_length)
    per_scale = np.round(len(time_unique) / window_num)  # 每个窗口的时间间隔
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
    return adj_list
def newdata(args,node):
    train2 = np.loadtxt("../datanew/" + args.data + "/" + args.data + "_2-train.txt")
    adj_train_time,time_length_train = compare_pre_time_adj_train(args.windows, node, train2)
    return adj_train_time