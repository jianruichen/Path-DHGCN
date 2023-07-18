# Path-DHGCN
This code is for paper "Simplex pattern prediction based on dynamic
higher-order path graph convolutional networks", the modle named "Path-DHGCN'.

Envoriment:
pip install numpy
pip install pandas
pip install torch
pip install scipy
pip install matplotlib

Hyperparameters:
- `task`: The task of simplex pattern prediction :
- `data`: The dataset you want to run this model
- `epoch`: The total number of training.
- `lr`: The learning rate.
- `windows`: Size of the time window divided.
- `hidden`: Number of hidden units.
- `dropout`: drop out rate for input layer of models.

process(such as):
run "main.py":
-task triangles -data email-Eu  -lr 0.001 -windows 30 -hidden 128  -epoch 400 -dropout 0.7  




