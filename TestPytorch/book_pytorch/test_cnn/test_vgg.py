# -*- coding: utf-8 -*- 
"""
__version__ = v1.0
__file__ = test_vgg.py
__title__ = ''
__author__ = 'lb'
__mtime__ = 2018/5/2 15:26 
__des__=''
"""
import sys
sys.path.append('...')

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from utils import train

def vgg_block(num_convs,in_channels,out_channels):
    net = [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,padding=1),nn.ReLU(True)]

    for i in range(num_convs-1):
        net.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        net.append(nn.ReLU(True))

    net.append(nn.MaxPool2d(2,2))
    return nn.Sequential(*net)

block_demo = vgg_block(3,64,128)
print(block_demo)

imput_demo = Variable(torch.zeros(1,64,300,300))
output_demo = block_demo(imput_demo)
print(output_demo.shape)

def vgg_stack(num_convs,channels):
    net = []
    for n,c in zip(num_convs,channels):
        print(n)
        print(c)
        in_c = c[0]
        out_c = c[1]
        net.append(vgg_block(n,in_c,out_c))
    return nn.Sequential(*net)

vgg_net = vgg_stack((1,1,2,2,2),((3,64),(64,128),(128,256),(256,512),(512,512)))
print(vgg_net)

test_x = Variable(torch.zeros(1,3,256,256))
test_y = vgg_net(test_x)
print(test_y.shape)

class vgg(nn.Module):
    def __init__(self):
        super(vgg,self).__init__()
        self.feature = vgg_net
        self.fc = nn.Sequential(nn.Linear(512,100),nn.ReLU(True),nn.Linear(100,10))

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0],-1)
        x = self.fc(x)
        return x

def data_tf(x):
    x = np.array(x, dtype='float=32')
    x = (x-0.5)/0.5
    x = x.transpose((2,0,1))
    x = torch.from_numpy(x)
    return x

train_set = CIFAR10('./data',train=True,transform=data_tf,download=True)
train_data = torch.utils.data.DataLoader(train_set,batch_size = 64,shuffle=True)
test_set = CIFAR10('./data',train=False,transform=data_tf,download=True)
test_data = torch.utils.data.DataLoader(train_set,batch_size = 128,shuffle=False)

net = vgg()
optimizer = torch.optim.SGD(net.parameters(),lr = 1e-1)
criterion = nn.CrossEntropyLoss()

train(net,train_data,test_data,20,optimizer,criterion)