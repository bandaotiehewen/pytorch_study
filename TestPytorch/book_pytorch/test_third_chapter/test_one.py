# -*- coding: utf-8 -*-
"""
__version__ = v1.0
__file__ = test_one.py
__title__ = ''
__author__ = 'lb'
__mtime__ = 2018/4/18 10:00 
__des__=''
"""
import torch
from torch.autograd import Variable

# x = Variable(torch.Tensor([2]), requires_grad=True)
# y = x + 2
# z = y ** 2 + 3
# print(z)
#
# z.backward()
# print(x.grad)
#
# x = Variable(torch.randn(10, 20), requires_grad=True)
# y = Variable(torch.randn(10, 5), requires_grad=True)
# w = Variable(torch.randn(20, 5), requires_grad=True)
#
# out = torch.mean(y - torch.matmul(x, w))
# out.backward()
#
# print(x.grad)

# m = Variable(torch.FloatTensor([[2, 3]]), requires_grad=True)
# n = Variable(torch.zeros(1, 2))
# print(m)
# print(n)
#
# n[0, 0] = m[0, 0] ** 2
# n[0, 1] = m[0, 1] ** 3
# print(n)
#
# w = torch.ones_like(n)
# print(w)
# n.backward(w)
# print(m.grad)

# x = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
# k = Variable(torch.zeros(2))
#
# k[0] = x[0] ** 2 + 3 * x[1]
# k[1] = 2 * x[0] + x[1] ** 2
# print(k)
#
# j = Variable(torch.zeros(2, 2))
# print(j)
# k.backward(torch.FloatTensor([1, 0]), retain_graph=True)
# j[0] = x.grad.data
# print(j[0])
#
# x.grad.data.zero_()
# k.backward(torch.FloatTensor([0, 1]))
# j[1] = x.grad.data
# print(j[1])
#
# print(j)

x = torch.randn(3)
print(x)
x = Variable(x, requires_grad=True)
y = x * 2
print(y)

y.backward(torch.FloatTensor([1, 0.1, 0.01]))
print(x.grad)

