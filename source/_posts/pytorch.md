---
categories: Machine Learning
title: Pytorch 基本概念
date: 2019-12-09 15:08:03
tags: [Machine Learning, Python, Pytorch]
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

参考：[code of learn deep learning with pytroch](https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch/blob/master/chapter2_PyTorch-Basics/PyTorch-introduction.ipynb)

# 安装
conda: `conda install pytorch torchvision -c pytorch`
pip: `pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl`, `pip install torchvision`
[官网](https://pytorch.org/)

# Tensor & Variable
1. Pytorch 是一个拥有强力GPU加速的张量和动态构建网络的库，其主要构建是张量。
```
import torch
import numpy as np

# 创建 numpy ndarray
numpy_tensor = np.random.randn(10,20)

# 创建 pytorch tensor
pytorch_tensor = torch.randn(3,2)

# 将 ndarray 转换到 tensor 上
pytorch_tensor1 = torch.Tensor(numpy_tensor)
pytorch_tensor2 = torch.from_numpy(numpy_tensor)

# 将 tensor 转换为 ndarray
# cpu
numpy_array = pytorch_tensor1.numpy()
# gpu
numpy_array = pytorch_tensor1.cpu().numpy()
```

2. 将 tensor 放到 GPU 上
```
# 第一种方式：定义 cuda 数据类型
dtype = torch.cuda.FloatTensor # 定义默认 GPU 的 数据类型
gpu_tensor = torch.randn(10, 20).type(dtype)

# 第二种方式：
gpu_tensor = torch.randn(10, 20).cuda(0) # 将 tensor 放到第一个 GPU 上
gpu_tensor = torch.randn(10, 20).cuda(1) # 将 tensor 放到第二个 GPU 上
```
使用第一种方式将 tensor 放到 GPU 上的时候会将数据类型转换成定义的类型，而是用第二种方式能够直接将 tensor 放到 GPU 上，类型跟之前保持一致

3. 将 tensor 放回 CPU
```
cpu_tensor = gpu_tensor.cpu()
```

4. 访问 tensor 的属性
```
# 大小
pytorch_tensor1.shape
pytorch_tensor1.size()

# 类型
pytorch_tensor1.type()
pytorch_tensor1.type(torch.DoubleTensor) #转换为float64

# 维度
pytorch_tensor1.dim()

# 所有元素个数
pytorch_tensor1.numel()
```

5. Tensor 操作
```
x = torch.ones(2,2) # float tensor

x = x.long() # 转换为整型
x = x.float() # 转换为 float


x = torch.randn(4,3) # 随机矩阵

# 沿行取最大值，返回每一行最大值及下标
max_value, max_index = torch.max(x, dim=1) 

# 沿行求和
sum_x = torch.sum(x, dim=1)

# 增加维度
x = x.unsqueeze(0) # 在第一维加

# 减少维度
x = x.squeeze(0) # 减少第一维
x = x.squeeze() # 将 tensor 中所有一维去掉

# 维度交换
x = x.permute(1,0,2) # 重新排列 tensor 的维度
x = x.transpose(0,2) # 交换 tensor 中的两个维度

# 使用 view 对 tensor 进行 reshape
x = torch.randn(3,4,5)
x = x.view(-1,5) # -1 表示任意大小，5表示第二维变成5

# 两个 tensor 求和
x = torch.randn(3,4)
y = torch.randn(3,4)

z = x+y
z = torch.add(x,y)

# inplace 操作，在操作的符号后加_
x.unsqueeze_(0)
x.transpose_(1,0)
x.add_(y)
```

6. Variable
Variable 是对 tensor 的封装，包含三个属性：`.data` tensor 本身，`.grad` tensor 的梯度，`.grad_fn` variable 的获得方式。
```
from torch.autograd import Variable

x_tensor = torch.randn(10, 5)
y_tensor = torch.randn(10, 5)

# 将 tensor 变成 Variable
x = Variable(x_tensor, requires_grad=True) # 默认 Variable 是不需要求梯度
y = Variable(y_tensor, requires_grad=True)

z = torch.sum(x + y)

print(z.data)
print(z.grad_fn)

# 求 x 和 y 的梯度
z.backward()

print(x.grad)
print(y.grad)
```

# 自动求导
1. 简单情况
```
import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([2]), requires_grad=True)
y = x + 2
z = y ** 2 + 3

z.backward()
print(x.grad)
```

2. 复杂情况
```
m = Variable(torch.FloatTensor([[2, 3]]), requires_grad=True) # 构建一个 1 x 2 的矩阵
n = Variable(torch.zeros(1, 2)) # 构建一个相同大小的 0 矩阵

n[0, 0] = m[0, 0] ** 2
n[0, 1] = m[0, 1] ** 3

n.backward(torch.ones_like(n)) # 将 (w0, w1) 取成 (1, 1)
print(m.grad)
```

3. 多次自动求导
```
x = Variable(torch.FloatTensor([3]), requires_grad=True)
y = x * 2 + x ** 2 + 3

y.backward(retain_graph=True) # 设置 retain_graph 为 True 来保留计算图
print(x.grad) # 8

y.backward() # 再做一次自动求导，这次不保留计算图
print(x.grad) # 16

```
这里做了两次自动求导，16 为第一次的梯度 8 和第二次的梯度 8 加和结果。

4. 练习
```
x = Variable(torch.FloatTensor([2, 3]), requires_grad=True)
k = Variable(torch.zeros(2))

k[0] = x[0] ** 2 + 3 * x[1]
k[1] = x[1] ** 2 + 2 * x[0]

j = torch.zeros(2, 2)

k.backward(torch.FloatTensor([1, 0]), retain_graph=True)
j[0] = x.grad.data

x.grad.data.zero_() # 归零之前求得的梯度

k.backward(torch.FloatTensor([0, 1]))
j[1] = x.grad.data
```

# 线性模型和梯度下降
```
import torch
import numpy as np
from torch.autograd import Variable

torch.manual_seed(2017)

# 读入数据 x 和 y
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 画出图像
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(x_train, y_train, 'bo')

# 转换成 Tensor
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 定义参数 w 和 b
w = Variable(torch.randn(1), requires_grad=True) # 随机初始化
b = Variable(torch.zeros(1), requires_grad=True) # 使用 0 进行初始化

# 构建线性回归模型
x_train = Variable(x_train)
y_train = Variable(y_train)

def linear_model(x):
    return x * w + b

y_ = linear_model(x_train)

# 模型输出
plt.plot(x_train.data.numpy(), y_train.data.numpy(), 'bo', label='real')
plt.plot(x_train.data.numpy(), y_.data.numpy(), 'ro', label='estimated')
plt.legend()

# 计算误差
def get_loss(y_, y):
    return torch.mean((y_ - y_train) ** 2)

loss = get_loss(y_, y_train)

# 自动求导
loss.backward()

# 更新一次参数
w.data = w.data - 1e-2 * w.grad.data
b.data = b.data - 1e-2 * b.grad.data

# 进行 10 次更新
for e in range(10):
    y_ = linear_model(x_train)
    loss = get_loss(y_, y_train)
    
    w.grad.zero_() # 记得归零梯度
    b.grad.zero_() # 记得归零梯度
    loss.backward()
    
    w.data = w.data - 1e-2 * w.grad.data # 更新 w
    b.data = b.data - 1e-2 * b.grad.data # 更新 b 
    print('epoch: {}, loss: {}'.format(e, loss.data[0]))
```

1. 获得 `[x,x^2,x^3]`：
```
x_sample = np.arange(-3, 3.1, 0.1)
x_train = np.stack([x_sample ** i for i in range(1, 4)], axis=1)
```
2. 线性模型
```
def multi_linear(x):
    return torch.mm(x, w) + b
```

# Logistic 回归
```
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


# 设定随机种子
torch.manual_seed(2017)

# 从 data.txt 中读入点
with open('./data.txt', 'r') as f:
    data_list = [i.split('\n')[0].split(',') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]

# 标准化
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0]/x0_max, i[1]/x1_max, i[2]) for i in data]

x0 = list(filter(lambda x: x[-1] == 0.0, data)) # 选择第一类的点
x1 = list(filter(lambda x: x[-1] == 1.0, data)) # 选择第二类的点

np_data = np.array(data, dtype='float32') # 转换成 numpy array
x_data = torch.from_numpy(np_data[:, 0:2]) # 转换成 Tensor, 大小是 [100, 2]
y_data = torch.from_numpy(np_data[:, -1]).unsqueeze(1) # 转换成 Tensor，大小是 [100, 1]

# 定义 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

import torch.nn.functional as F

# 计算loss
def binary_loss(y_pred, y):
    logits = (y * y_pred.clamp(1e-12).log() + (1 - y) * (1 - y_pred).clamp(1e-12).log()).mean() # clamp 约束最大值最小值
    return -logits

# 使用 torch.optim 更新参数
from torch import nn
w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))

def logistic_regression(x):
    return F.sigmoid(torch.mm(x, w) + b)

optimizer = torch.optim.SGD([w, b], lr=1.)

# 进行 1000 次更新
import time

start = time.time()
for e in range(1000):
    # 前向传播
    y_pred = logistic_regression(x_data)
    loss = binary_loss(y_pred, y_data) # 计算 loss
    # 反向传播
    optimizer.zero_grad() # 使用优化器将梯度归 0
    loss.backward()
    optimizer.step() # 使用优化器来更新参数
    # 计算正确率
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().data[0] / y_data.shape[0]
    if (e + 1) % 200 == 0:
        print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(e+1, loss.data[0], acc))
during = time.time() - start
print()
print('During Time: {:.3f} s'.format(during))
```
- pytorch 中包含一些常见 [loss](https://pytorch.org/docs/0.3.0/nn.html#loss-functions)，如线性回归分类`nn.MSE()`和二分类`nn.BCEWithLogitsLoss()`
```
# 使用自带的loss
criterion = nn.BCEWithLogitsLoss() # 将 sigmoid 和 loss 写在一层，有更快的速度、更好的稳定性

w = nn.Parameter(torch.randn(2, 1))
b = nn.Parameter(torch.zeros(1))

def logistic_reg(x):
    return torch.mm(x, w) + b

optimizer = torch.optim.SGD([w, b], 1.)

y_pred = logistic_reg(x_data)
loss = criterion(y_pred, y_data)
print(loss.data)

# 同样进行 1000 次更新
start = time.time()
for e in range(1000):
    # 前向传播
    y_pred = logistic_reg(x_data)
    loss = criterion(y_pred, y_data)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # 计算正确率
    mask = y_pred.ge(0.5).float()
    acc = (mask == y_data).sum().data / y_data.shape[0]
    if (e + 1) % 200 == 0:
        print('epoch: {}, Loss: {:.5f}, Acc: {:.5f}'.format(e+1, loss.data, acc))

during = time.time() - start
print()
print('During Time: {:.3f} s'.format(during))
```