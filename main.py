from __future__ import division
from __future__ import print_function
from numpy import *
import torch.utils.data as Data

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from pygcn.utils import load_data, norm,accuracy,newdata,features_fusion3,features_fusion4,features_fusion5
from pygcn.model import HTGCN,MLP
import warnings
warnings.filterwarnings("ignore")
# Training settings
parser = argparse.ArgumentParser()#参数
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')#随机种子
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--windows', type=int, default=30,
                    help='Number of time windows.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--task', type=str, help='task',
                    choices=['triangles','quads','pentagon'],
                    default='triangles')
parser.add_argument('-d', '--data', type=str,help='data sources to use',
                    choices=['DAWN', 'tags-ask-ubuntu', 'tags_math_sx', 'NDC-classes', 'congress-bills',
                             'threads-ask-ubuntu', 'email-Eu','contact-primary-school'],
                    default='congress-bills')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
if args.data == "tags_math_sx":
    node = 1627
elif args.data == "congress-bills":
    node = 1718
elif args.data == "contact-primary-school":
    node = 242
elif args.data == "email-Eu":
    node = 998
elif args.data == "email-Enron":
    node = 143
elif args.data == "DAWN":
    node = 2560
elif args.data == "tags-ask-ubuntu":
    node = 3029
elif args.data == "threads-ask-ubuntu":
    node = 125602
alltime,train_list,train_neg_list,val_list,val_neg_list,test_list,test_neg_list=load_data(args,node)
#train_adj,val_adj, test_adj=ADJ(args)
train_adj_list=newdata(args,node)
features = torch.nn.Embedding(node, 256)  # 初始特征
features = features.weight
features = nn.init.xavier_uniform_(features, gain=1)
# Model and optimizer
model =HTGCN(nfeat=features.shape[1],#动态高阶时序GCN
            nhid=args.hidden,
            dropout=args.dropout,
            dimension=64)
discrim=MLP(dropout=args.dropout)#高阶结构鉴别器
if args.cuda:
    model.cuda()
    discrim.cuda()
    features = features.cuda()
def loss(output,sign):#损失函数
    criterion1 = torch.nn.BCELoss()#二元交叉熵损失函数
    output = output.squeeze(-1)
    output = output.float()
    sign = sign.float()
    loss_term = criterion1(output,sign)  # 将预测的与真实值进行损失计算
    reg_loss = (1 / 2) * (((model.gc1.weight.norm(2)) + (model.gc2.weight.norm(2))+model.w.weight.norm(2)))  # + (model.gc3.weight.norm(2))
    reg_loss = reg_loss * args.weight_decay
    loss_term=loss_term+reg_loss
    return loss_term

def train(epoch):#开始训练
    t = time.time()
    model.train()#加载训练模型
    discrim.train()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)  # 梯度更新，优化
    optimizer_discrim = optim.Adam(discrim.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    optimizer_discrim.zero_grad()
    train_shots = list(range(0, len(train_adj_list)-1))
    loss_train_epoch=[]
    train_AUC_epoch=[]
    train_AUC_PR_epoch=[]
    for t in train_shots:
       train_adj_time=norm(train_adj_list[t])
       if args.task=="triangles":
           node_embedding = model.forward1(args,features, train_adj_time)
           time_embedding = model.forward2(alltime)
           all_emb, labels_all,time_emb = features_fusion3(node_embedding, time_embedding, alltime, train_list,train_neg_list)
       elif args.task == "quads":
           node_embedding = model.forward1(args,features, train_adj_time)
           time_embedding = model.forward2(alltime)
           all_emb, labels_all, time_emb = features_fusion4(node_embedding, time_embedding, alltime,train_list,train_neg_list)
       elif args.task == "pentagon":
           node_embedding = model.forward1(args, features, train_adj_time)
           time_embedding = model.forward2(alltime)
           all_emb, labels_all, time_emb = features_fusion5(node_embedding, time_embedding, alltime,train_list,train_neg_list)
       output=discrim(all_emb)
       loss_train=loss(output,labels_all)
       loss_train.backward()
       optimizer.step()
       optimizer_discrim.step()
       loss_train_epoch.append(loss_train.item())
       output=torch.tensor(output)
       train_AUC,train_AUC_PR=accuracy(labels_all,output)
       train_AUC_epoch.append(train_AUC)
       train_AUC_PR_epoch.append(train_AUC_PR)
    model.eval()
    loss_train=np.mean(loss_train_epoch)
    train_AUC=np.mean(train_AUC_epoch)
    train_AUC_PR=np.mean(train_AUC_PR_epoch)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'train_AUC: {:.4f}'.format(train_AUC.item()),
          'train_AUC_PR: {:.4f}'.format(train_AUC_PR.item()),
          )
    return node_embedding,time_embedding
def val(node_embedding,time_embedding):
    model.eval()
    discrim.eval()
    if args.task == "triangles":
        all_emb_val, labels_all_val, time_emb = features_fusion3(node_embedding, time_embedding, alltime, val_list,val_neg_list)
    elif args.task == "quads":
        all_emb_val, labels_all_val, time_emb = features_fusion4(node_embedding, time_embedding, alltime, val_list,val_neg_list)
    elif args.task == "pentagon":
        all_emb_val, labels_all_val, time_emb = features_fusion5(node_embedding, time_embedding, alltime, val_list,val_neg_list)
    output_val = discrim(all_emb_val)
    loss_val = loss(output_val, labels_all_val)
    output_val = torch.tensor(output_val)
    val_AUC, val_AUC_PR = accuracy(labels_all_val, output_val)

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'val_AUC: {:.4f}'.format(val_AUC.item()),
          'val_AUC_PR: {:.4f}'.format(val_AUC_PR.item()))
    return val_AUC,val_AUC_PR
def test(node_embedding,time_embedding):
    model.eval()
    discrim.eval()
    if args.task == "triangles":
        all_emb_test,labels_test,time_emb = features_fusion3(node_embedding, time_embedding, alltime, test_list,test_neg_list)
    elif args.task == "quads":
        all_emb_test, labels_test, time_emb = features_fusion4(node_embedding, time_embedding, alltime, test_list,test_neg_list)
    elif args.task == "pentagon":
        all_emb_test, labels_test, time_emb = features_fusion5(node_embedding, time_embedding, alltime, test_list,test_neg_list)
    output_test = discrim(all_emb_test)
    loss_test = loss(output_test, labels_test)
    output_test=torch.tensor(output_test)
    test_AUC, test_AUC_PR = accuracy(labels_test,output_test)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "test_AUC= {:.4f}".format(test_AUC.item()),
          "test_AUC_PR= {:.4f}".format(test_AUC_PR.item()),
      )
    return test_AUC, test_AUC_PR

# Train model
val_auc=[]
val_auc_pr=[]
test_auc=[]
test_auc_pr=[]
for epoch in range(args.epochs):
    node_embedding, time_embedding = train(epoch)
    if epoch % 10 == 0:  # 十次测试一次
        print("[TEST]")
        test_AUC,test_AUC_PR=test(node_embedding,time_embedding)
        val_AUC,val_AUC_PR=val(node_embedding,time_embedding)
        test_auc.append(test_AUC)
        test_auc_pr.append(test_AUC_PR)
        val_auc.append(val_AUC)
        val_auc_pr.append(val_AUC_PR)
print("Optimization Finished!")
a=max(test_auc)
b=max(test_auc_pr)
c=max(val_auc)
d=max(val_auc_pr)
print("The best AUC",a)
print("The best AUC_PR",b)
print("The best AUC",c)
print("The best AUC_PR",d)
# Testing
AUC,AUC_P=test(node_embedding,time_embedding)
