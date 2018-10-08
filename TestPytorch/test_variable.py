from torch.autograd import Variable
import torch
from torch import nn

# x_tensor = torch.randn(10,5)
# y_tensor = torch.randn(10,5)
# print(x_tensor)
# print(y_tensor)
#
# x = Variable(x_tensor,requires_grad=True)
# y = Variable(y_tensor,requires_grad=True)
#
# z = torch.sum(x + y)
# print(z.data)
# print(z.grad_fn)
#
# z.backward()
# print(x.grad)
# print(y.grad)

import matplotlib.pyplot as plt
import numpy as np
# x = np.arange(-3,3.01,0.1)
# y = x**2
# plt.plot(x,y)
# plt.plot(2,4,'ro')
# plt.show()
#
# x = Variable(torch.FloatTensor([2]),requires_grad=True)
# y = x**2
# y.backward()
# print(x.grad)

# net1 = nn.Sequential(
#     nn.Linear(30, 40),
#     nn.ReLU(),
#     nn.Linear(40, 50),
#     nn.ReLU(),
#     nn.Linear(50, 10)
# )
#
# w1 = net1[0].weight
# b1 = net1[0].bias
#
# print(w1,w1.size())
# print(b1,b1.size())


# x = torch.arange(15).view(5, 3)

# print(x.type())
# print(x.float())

# x = x.float()

# print(x.shape)
# print(x.size())
# print(x.shape[0],x.shape[1])
# print(x.size(0),x.size(1))

# gamma = torch.ones(x.shape[1])
# beta = torch.zeros(x.shape[1])
# print(gamma)
# print(beta)

# def simple_batch_norm_1d(x, gamma, beta):
#     eps = 1e-5
#     x_mean = torch.mean(x, dim=0, keepdim=True) # 保留维度进行 broadcast
#     print(x_mean)
#     x_var = torch.mean((x - x_mean) ** 2, dim=0, keepdim=True)
#     print(x_var)
#     x_hat = (x - x_mean) / torch.sqrt(x_var + eps)
#     print(x_hat,x_hat.size())
#     print(gamma.view_as(x_mean),gamma.view_as(x_mean).size())
#     print(gamma.view_as(x_mean) * x_hat)
#     return gamma.view_as(x_mean) * x_hat + beta.view_as(x_mean)

# print('before bn: ')
# print(x)
# y = simple_batch_norm_1d(x, gamma, beta)
# print('after bn: ')
# print(y)


a = torch.Tensor([[1,2],[3,4],[5,6]])
print(a)
b = torch.Tensor([[1,2]])

print(b)
c = a*b
print(c)

d = torch.Tensor([[1],[2],[3]])
print(d)
e = a*d
print(e)