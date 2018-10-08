# -*- coding: utf-8 -*-
"""
__version__ = v1.0
__file__ = test_basic.py
__title__ = ''
__author__ = 'lb'
__mtime__ = 2018/8/24 21:37 
__des__=''
"""
from torch.autograd import Variable
# word = "when forty winters shall besiege thy brow, And dig deeo trenches in the beauty's filed, The youth's proud livery".split()
# # print(word)
# vocd = set(word)
# # print(vocd)
# word_to_idx = {word: i for i, word in enumerate(vocd)}
# print(word_to_idx)
# idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
# print(idx_to_word)
#
# training_data = [("The dog ate the apple".split(),["DET","NN","V","DET","NN"]),
#                  ("Everybody read that book".split(),["NN","V","DET","NN"])]
# print(training_data)

# word_to_idx = {}
# tag_to_idx = {}
# for context,tag in training_data:
#     print("context:"+str(context))
#     print("tag:" + str(tag))
#     for word in context:
#         if word not in word_to_idx:
#             word_to_idx[word] = len(word_to_idx)
#     print("word_to_idx:"+str(word_to_idx))
#     for lable in tag:
#         if lable not in tag_to_idx:
#             tag_to_idx[lable] = len(tag_to_idx)
#     print("tag_to_idx:"+str(tag_to_idx))

import torch
# a = torch.Tensor(2,3)
# print(a)
# a = a.view(1,-1)
# print(a)
#
# b = torch.Tensor(1,3)
# print(b)
# print(b.squeeze(0))
# print(b.squeeze(1))
#
# c = torch.Tensor(3)
# print(c)
# print(c.unsqueeze(1))

# v_data = [1.,2.,3.]
# V = torch.Tensor(v_data)
# print(V)
#
# M_data = [[1., 2., 3.], [4., 5., 6.]]
# M = torch.Tensor(M_data)
# print(M)
#
# T_data = [[[1.,2.], [3.,4.]],
#           [[5.,6.], [7.,8.]]]
# T = torch.Tensor(T_data)
# print(T)
#
# print(V[0])
# print(M[0])
# print(T[0])
#
# # x = torch.randn((3,4,5))
# # print(x)
#
# x = torch.Tensor([1., 2., 3.])
# y = torch.Tensor([4., 5., 6.])
# z = x + y
# print(z)


# x_1 = torch.randn(2, 5)
# print(x_1)
# y_1 = torch.randn(3, 5)
# print(y_1)
# z_1 =torch.cat([x_1, y_1])#没有最后一个参数，默认是0，则最终维度的第0维度为x_1与y_1第0维度的和，最终维度的其他维度不变.以下同理
# print(z_1)
#
# x_2 = torch.randn(3, 3)
# print(x_2)
# y_2 = torch.randn(3, 5)
# print(y_2)
# z_2 = torch.cat([x_2, y_2], 1)
# print(z_2)

# x = torch.randn(2, 3, 4)
# print(x)
# print(x.dim())
#
# print(x.view(2,12))#将234 -> 2*12
# x = x.view(2,12)
# print(x.dim())
#
# print(x.view(2,-1))#-1的话，意味着最后的相乘为维数，这里为2*之后的成绩
# print(x.view(-1,2))
# print(x.view(-1))
#
# x_1 = torch.rand(5,5)
# print(x_1)
# print(x_1.view(-1,5*5))
# print(x_1.view(5*5,-1))
#
# print(x_1.view(-1,5))
# print(x_1.view(5,-1))
#
# y = torch.randn(5,3)
# print(y)
# z = y.view(3,-1)
# print(z)

import torch.autograd as autograd

# x = autograd.Variable(torch.Tensor([1., 2., 3]), requires_grad=True)
# print(x)
# print(x.data)#.data显示具体数据
#
# y = autograd.Variable( torch.Tensor([4., 5., 6]), requires_grad=True )
# z = x + y
# print(z.data)

# s = z.sum()
# print(s)

# s.backward()
# print(x.grad)

# x_tensor = torch.randn(10, 5)
# y_tensor = torch.randn(10, 5)
#
# # 将 tensor 变成 Variable
# x = Variable(x_tensor, requires_grad=True) # 默认 Variable 是不需要求梯度的，所以我们用这个方式申明需要对其进行求梯度
# y = Variable(y_tensor, requires_grad=True)
#
# z = x + y
# print(z)
# print(z.data[0])
# z = torch.sum(z)
# print(z)
#
# print(z.data)
# print(z.grad_fn)

import torch.nn as nn
# lin = nn.Linear(5,3)
# input_data = torch.randn(2,5)
# print(input_data)
# data = autograd.Variable(input_data)
# output_data = lin(data)
# print(output_data)

import torch.nn.functional as F
# input_data = torch.randn(2, 2)
# print(input_data)
# data = autograd.Variable( input_data )
# print(data)
# print (F.relu(data))#relu函数是小于零是0，大于零就是它本身

# data = autograd.Variable( torch.randn(5) )
# print(data)
# print(F.softmax(data))
# print(F.softmax(data).sum())
# print(F.log_softmax(data))

# b=torch.Tensor(3,1)
# print(b)
# print(b.squeeze(0))
# print(b.squeeze(1))
# print(b.squeeze())
#
# c=torch.Tensor(3)
# print(c)
# print(c.unsqueeze(0))
# print(c.unsqueeze(1))
#
# d = torch.rand(4, 1, 3)
# print(d)
# print(d.size())
# print(d.squeeze())


# d=torch.Tensor([[1,3],[2,4]])
# print(d)
# print(torch.max(d,0))
# print(torch.max(d,1))


# from torch.autograd import Variable
# x = Variable(torch.ones(2, 2), requires_grad = True)
# print(x)
# y = x + 2
# print(y)
#
# # y 是作为一个操作的结果创建的因此y有一个creator
# z = y * y * 3
# print(z)
# out = z.mean()
# print(out)
#
# # 现在我们来使用反向传播
# out.backward()
#
# # out.backward()和操作out.backward(torch.Tensor([1.0]))是等价的
# # 在此处输出 d(out)/dx
# print(x.grad)

# x = torch.randn(3)
# x = Variable(x, requires_grad = True)
# y = x * 2
# while y.data.norm() < 1000:
#     y = y * 2
# gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
# y.backward(gradients)
# print(x.grad)

# from torch import nn, optim
# class Cnn(nn.Module):
#     def __init__(self):
#         super(Cnn, self).__init__()
#         print('init')
#
#     def forward(self, x):
#         print('111111111111111')
#         print(x)
#         print('222222222222222')
#         return x
#
#     def backward(self, y):
#         print('333333333333333')
#         return y
#
# cnn = Cnn()
# cnn('hhhhh')
# cnn.forward('aaaa')

#
# x = torch.randn(3,2)
# print(x)
# print(x.shape)
# print(x.unsqueeze(0))
# print(x.unsqueeze(0).shape)
# #
# print(x.unsqueeze(1))
# print(x.unsqueeze(1).shape)
#
# print(x.squeeze(0))
# print(x.squeeze(0).shape)
#
# # max_value, max_idx = torch.max(x,dim=0)
# # print(max_value)
# # print(max_idx)
#
# # sum_x = torch.sum(x,dim=1)
# # print(sum_x)
# # print(sum_x.shape)
#
# y = torch.randn(4,3,2)
# print(y)
# print(y.shape)
# print(y.squeeze(0))
# print(y.squeeze(0).shape)

# x = torch.randn(4, 3)
# print(x)
#
# print(x.shape)
# x = x.unsqueeze(0) # 在第一维增加
# print(x)
# print(x.shape)
#
# x = x.unsqueeze(1) # 在第二维增加
# print(x)
# print(x.shape)
#
# x = x.squeeze(2) # 减少第一维
# print(x.shape)
#
# x = x.squeeze() # 将 tensor 中所有的一维全部都去掉
# print(x)
# print(x.shape)


# x = torch.randn(3, 4, 5)
# print(x)
# print(x.shape)
#
# # 使用permute和transpose进行维度交换
# x = x.permute(1, 0, 2) # permute 可以重新排列 tensor 的维度
# print(x)
# print(x.shape)
#
# x = x.transpose(0, 2)  # transpose 交换 tensor 中的两个维度
# print(x)
# print(x.shape)


# # 使用 view 对 tensor 进行 reshape
# x = torch.randn(3, 4, 5)
# print(x.shape)
#
# x = x.view(-1, 5) # -1 表示任意的大小，5 表示第二维变成 5
# print(x.shape)
#
# x = x.view(3, 20) # 重新 reshape 成 (3, 20) 的大小
# print(x.shape)


# x = torch.ones(3, 3)
# print(x.shape)
#
# # unsqueeze 进行 inplace
# x.unsqueeze_(0)
# print(x.shape)
#
# # transpose 进行 inplace
# x.transpose_(1, 0)
# print(x.shape)

# x = torch.ones(4,4)
# print(x.type())

# x = torch.arange(1,10,1)
# print(x)


# b = torch.Tensor([[1,2,3,],[4, 5,6]])
# print(b)
# print(b.size())
# print(b.size(0))
# c = b.tolist()
# print(c.size())

# 创建一个和b形状一样的tesor
# b_size = b.size()
# c = torch.Tensor(b_size)
# print(c)
# # 创建一个元素为2和3的tensor
# d = torch.Tensor((2, 3))
# print(d)
# print(d.dim())

# a = torch.arange(0, 6)
# a= a.view(2, 3)
# print(a)
#
# b = a.unsqueeze(1)
# print(b)
# print(b.size())
#
# c = b.view(1, 1, 1, 2, 3)
# print(c)
# print(c.squeeze())

# a = torch.randn(3,4)
# print(a)
#
# print(a[0:1])
# print(a[:,2])
# print(a[:,:2])
#
# print(a[0:1,:2])
# print(a[0,:2])
# print(a>1)
#
# print(a[torch.LongTensor([0,1])])


# a = torch.arange(0, 16).view(4, 4)
# index = torch.LongTensor([[0,1,2,3]])
# print(a.gather(1,index))
# b = torch.randn(2,3)
# print(b)
# c = b.sum(dim = 0,keepdim = False)
# print(c)
# d = b.sum(dim = 1,keepdim = False)
# print(d)

import numpy as np

# a = np.ones([2,3],dtype=np.float32)
# print(a)
#
# b = np.ones([2, 3])
# print(b)

# a = torch.arange(0, 6)
# print(a.storage())
#
# a[1] = 100
#
# c = a[2:]
# print(c)
# print(c.storage())

# def for_loop_add(x,y):
#     result = []
#     for i,j in zip(x,y):
#         result = x + y
#     return result
#
# x = torch.zeros(100)
# y = torch.ones(100)

# a = torch.arange(0, 200000000)
# print(a)
# print(a[-1], a[-2]) # 32bit的IntTensor精度有限导致溢出
# b = torch.LongTensor()
# torch.arange(0, 2000000000, out=b) # 64bit的LongTensor不会溢出
# print(b[-1],b[-2])

# a = torch.arange(0,10)
# print(a)
#
# b = torch.arange(5,15)
# print(b)
#
# print((a == b).sum())
#


# c = torch.IntTensor([0, 0, 0, 8, 5, 0, 0, 3, 8, 8, 5, 0, 0, 3, 0, 0, 0, 0, 3, 0, 8, 8, 8, 8,
#         3, 7, 0, 8, 5, 5, 8, 0])
# print(c)
# d = torch.IntTensor([4, 2, 3, 2, 8, 3, 8, 4, 0, 6, 5, 3, 2, 4, 5, 3, 0, 1, 0, 4, 9, 7, 6, 7,
#         7, 0, 3, 7, 4, 8, 9, 5])
# print(d)
# e = (c == d)
# print(e)
# print(e.sum())
# print(e.float().mean())

# f = c.eq(d)
# print(f)
# print(f.sum())

# a = torch.arange(2, 3)
# print(a)
# a.resize_(1, 3)
# print(a)
# a.resize_(3, 3)
# print(a)


# a = torch.arange(6).view(2, 3)
# print(a)
# print(a.sum())
# print(a.sum(dim=0, keepdim=True))
# print(a.sum(dim=0))


# a = torch.linspace(0,20,7)
# print(a)