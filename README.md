This code is for paper "Simplex pattern prediction based on dynamic higher-order path graph convolutional networks", the modle named "Path-DHGCN'.

Overview:

Recently, higher-order patterns have played an important role in network structure analysis. The simplices in higher-order patterns enrich dynamic network modeling and provide strong structural feature information for feature learning. However, the disorder dynamic network with simplex has not been effectively organized and divided, and current methods do not make full use of the feature information of simplex to predict the simplex pattern. To address these issues, we propose a simplex pattern prediction method based on dynamic higher-order path convolutional networks. Firstly, we divide the dynamic higher-order datasets into different network structures under continuous time windows, which possess complete time information. Secondly, feature extraction is performed on the network structure of continuous time windows through higher-order path convolutional networks. Subsequently, we embed time nodes into feature encoding and obtain feature representations of simplex patterns through feature fusion. The obtained feature representations of simplices are recognised by a simplex pattern discriminator to predict the simplex patterns at different moments. Finally, compared with other dynamic graph representation learning algorithms, our proposed algorithm has significantly improved its performance in predicting simplex patterns on five real dynamic higher-order datasets.

Envoriment: pip install numpy

pip install pandas

pip install torch

pip install scipy

pip install matplotlib

Settings:

task: The task of simplex pattern prediction :
data: The dataset you want to run this model
epoch: The total number of training.
lr: The learning rate.
windows: Size of the time window divided.
hidden: Number of hidden units.
dropout: drop out rate for input layer of models.
process(such as): run "main.py": -task triangles -data email-Eu -lr 0.001 -windows 30 -hidden 128 -epoch 400 -dropout 0.7
