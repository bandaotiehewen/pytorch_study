import numpy as np
import torch
from torchvision.datasets import mnist
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable

train_set = mnist.MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set = mnist.MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

a_data, a_lable = train_set[0]
# print(a_data)
print(a_lable)

# a_data = np.array(a_data,dtype='float32')
print(a_data.size())
print(a_data.size(0))

train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=64, shuffle=False)

b, b_lable = next(iter(train_data))
print(b.size())
print(b_lable.shape)
# print(type(b_lable))

net = nn.Sequential(
    nn.Linear(784, 400),
    nn.ReLU(),
    nn.Linear(400, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), 1e-1)

# 开始训练
losses = []
acces = []
eval_losses = []
eval_acces = []

i = 1
for e in range(20):
    train_loss = 0
    train_acc = 0
    net.train()

    # print(i)
    # i += 1
    # j = 1
    for im, label in train_data:
        # print(im.shape)
        # print(label.shape)

        # print(j)
        # j += 1


        im = im.view(im.size(0),-1)
        # print(im.size())
        im = Variable(im)
        # print(im)
        label = Variable(label)
        # print(label)
        # 前向传播
        out = net(im)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))


    # 在测试集上检验效果
    eval_loss = 0
    eval_acc = 0
    net.eval()  # 将模型改为预测模式
    for im, label in test_data:
        im = im.view(im.size(0),-1)
        im = Variable(im)
        label = Variable(label)
        out = net(im)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()
        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / im.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_data))
    print("eval_losses： " ,eval_losses)
    eval_acces.append(eval_acc / len(test_data))
    # print(eval_acces)
    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
          .format(e, train_loss / len(train_data), train_acc / len(train_data),
                  eval_loss / len(test_data), eval_acc / len(test_data)))
