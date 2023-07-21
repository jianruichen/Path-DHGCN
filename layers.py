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
