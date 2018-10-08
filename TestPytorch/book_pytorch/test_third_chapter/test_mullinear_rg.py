# -*- coding: utf-8 -*- 
"""
__version__ = v1.0
__file__ = test_mullinear_rg.py
__title__ = ''
__author__ = 'lb'
__mtime__ = 2018/4/18 16:03 
__des__=''
"""
import torch
from torch.autograd import Variable
from torch import nn,optim


def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, 4)], 1)


w_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)
b_garget = torch.FloatTensor([0.9])


def f(x):
    return x.mm(w_target) + b_garget[0]


def get_batch(batch_size=32):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)

    return Variable(x), Variable(y)

class ploy_model(nn.Module):
    def __init__(self):
        super(ploy_model,self).__init__()
        self.ploy = nn.Linear(3,1)

    def forward(self, x):
        out = self.ploy(x)
        return out

model = ploy_model()
cirterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr = 1e-3)

epoch = 0
while True:
    batch_x,batch_y = get_batch()
    output = model(batch_x)
    loss = cirterion(output,batch_y)
    print_loss = loss.data[0]
    print(print_loss )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch += 1
    print('epoch'+str(epoch))
    if print_loss < 1e-3:
        break
