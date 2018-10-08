# -*- coding: utf-8 -*- 
"""
__version__ = v1.0
__file__ = test_tutorials.py
__title__ = ''
__author__ = 'lb'
__mtime__ = 2018/4/12 20:23 
__des__=''
"""
from __future__ import print_function
import torch
import time

import numpy as np

# x = torch.Tensor(5, 4)
# print(x)
# x = torch.rand(5, 4)
# print(x)
# print(x.size())
#
# x = np.ones((5, 4))
# print(x)

# x = torch.rand(5,4)
# print(x)
# y = x.numpy()
# print(y)

# a = np.array([[3, 4], [3, 6]])
# print(a)
# b = torch.from_numpy(a)
# print(b)
#
# print(torch.cuda.is_available())

print(time.time())
a = torch.rand(5, 4)
print(a)
print(time.time())
a = a.cuda()
print(a)
print(time.time())