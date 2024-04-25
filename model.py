import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution,GraphAttentionLayer
import scipy.sparse as sp

class HTGCN(nn.Module):#GCN模型
    def __init__(self, nfeat, nhid, dropout,dimension):
        super(HTGCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, 64)
        self.dropout = dropout
        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)
        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())
    def forward1(self, args,x, adj):
           x = F.relu(self.gc1(x, adj))#得到经过计算的特征向量
           adj1 = adj.to_dense()
           adj1 = torch.matmul(adj1, adj1)
           x = F.dropout(x, self.dropout, training=self.training)#经过丢失率
           x=self.gc2(x,adj1)
           return x
    def forward2(self, alltime):
        t = np.array(alltime)
        t = t.astype(float)
        timestamps = torch.unsqueeze(torch.from_numpy(t).float(), dim=1)  # 维度扩充
        timestamps = timestamps.unsqueeze(dim=2)
        output = torch.cos(self.w(timestamps))
        output = output.squeeze(1)
        output = F.relu(output)
        return output
def sparse_mx_to_torch_sparse_tensor(sparse_mx):#归一化函数
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)#将矩阵转换成向量
class MLP(nn.Module):
    def __init__(self, dropout):
        super(MLP, self).__init__()
        self.task2_1 = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU()
        )
        self.task2_2 = nn.Sequential(
            nn.Linear(64, 16),
        )
        self.task2_3 = nn.Sequential(
            nn.Linear(16, 1),
        )
        self.task2_4 = nn.Sequential(
            nn.Linear(8, 1),
        )
        self.sm = nn.Sigmoid()  # 激活函数
    def forward(self, emb):
        out = self.task2_1(emb)
        out = self.task2_2(out)
        out = self.task2_3(out)
        out = self.sm(out)  # sigmoid激活，将值映射到【0,1】之间
        return out
