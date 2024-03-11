import math
import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
import torch.nn.functional as F
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features#输入特征
        self.out_features = out_features#输出特征
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#权重
        if bias:#偏移量
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))#初始化权重
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj1):
        support = torch.mm(input, self.weight)#输入乘以权重
        output = torch.spmm(adj1, support)#再乘以邻接矩阵
        #output=torch.spmm(adj2,output)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
class GraphAttentionLayer(nn.Module):#注意力网络层
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha,concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features#输入特征
        self.out_features = out_features#输出特征
        self.concat = concat
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))#权重参数，建立都是0
        nn.init.xavier_uniform_(self.W.data, gain=1.414)#初始化
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))#依然是一个权重参数
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)#非线性激活函数

    def forward(self, h, adj,adj_new):
        Wh = torch.mm(h, self.W) #特征乘以权重# h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)#得到每一对节点的注意力系数?

        zero_vec = -9e15*torch.ones_like(e)#？？？维度大小与e相同，所有元素都是-9*10的15次方
        attention = torch.where(adj > 0, e, zero_vec)#注意力矩阵
        attention_new=torch.where(adj_new > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)#将值映射到0-1之间
        attention = F.dropout(attention, self.dropout, training=self.training)#丢失率
        attention_new = F.softmax(attention_new, dim=1)#将值映射到0-1之间
        attention_new = F.dropout(attention_new, self.dropout, training=self.training)#丢失率
        attention=attention+attention_new
        h_prime = torch.matmul(attention, Wh)#注意力乘以权重，得到新的特征聚合后的表示？
        #h_prime_new = torch.matmul(attention_new, h_prime)
        if self.concat:
            return F.elu(h_prime),attention
        else:
            return h_prime
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])#自注意力指标？
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])#邻注意力指标？
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)#激活函数

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
