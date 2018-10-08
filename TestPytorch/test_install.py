# -*- coding: utf-8 -*- 
"""
__version__ = v1.0
__file__ = test_install.py
__title__ = ''
__author__ = 'lb'
__mtime__ = 2018/4/12 16:04 
__des__=''
"""

from __future__ import print_function
import torch
from torch.autograd import Variable


# print(torch.cuda.is_available())
# # x = torch.Tensor([[1,0],[2,5],[5,0]])
# x = torch.randn(3)
# x = Variable(x,requires_grad=True)
# print('a is:{}'.format(x))
# y = x*2
# print(y)
# y.backward(torch.FloatTensor([1,0.1,0.01]))
# print(x.grad)

# xx = x.cuda()
# print(xx)

# from torch.backends import cudnn
# print(cudnn.is_acceptable(xx))

def make_features(x):
    x = x.unsqueeze(1)
    print(x)
    return torch.cat([x**i for i in range(1,4)],1)
def f(x):
    return x.mm(w_target) + b_target[0]

w_target = torch.FloatTensor([1,1,1]).unsqueeze(1)
print(torch.FloatTensor([1,1,1]))
print(w_target)
b_target = torch.FloatTensor([0.9])
print(b_target)
print(b_target[0])


random = torch.randn(10)
print(random)
x = make_features(random)
print(x)
y = f(x)
print(y)