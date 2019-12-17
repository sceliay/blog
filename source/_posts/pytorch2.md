---
categories: Machine Learning
title: Pytorch 模型
date: 2019-12-17 18:59:45
tags: [Machine Learning, Python, Pytorch]
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

参考：[code of learn deep learning with pytroch](https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch/blob/master/chapter2_PyTorch-Basics/PyTorch-introduction.ipynb)

# Sequential 
Sequential 构建序列化模块。
```
seq_net = nn.Sequential(
	nn.Linear(2,4),
	nn.Tanh(),
	nn.Linear(4,1)
	)
```

- 序列模块可以通过索引访问每一层
```
seq_net[0] # 第一层
[out]: Linear(in_features=2, out_features=4)
```

- 打印权重
```
w0 = seq_net[0].weight
print(w0)
```

- 训练模型：
```
# 通过 parameters 可以取得模型的参数
param = seq_net.parameters()

# 定义优化器
optim = torch.optim.SGD(param, 1.)

# 训练 10000 次
for e in range(10000):
    out = seq_net(Variable(x))
    loss = criterion(out, Variable(y))
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (e + 1) % 1000 == 0:
        print('epoch: {}, loss: {}'.format(e+1, loss.data))
```

- 保存参数和模型
```
# 将参数和模型保存在一起
torch.save(seq_net, 'save_seq_net.pth')

# 读取保存的模型
seq_net1 = torch.load('save_seq_net.pth')
```

- 保存参数
```
# 保存模型参数
torch.save(seq_net.state_dict(), 'save_seq_net_params.pth')

# 先定义模型，再读取参数
seq_net2 = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)

seq_net2.load_state_dict(torch.load('save_seq_net_params.pth'))
```

# Module
更加灵活的模型定义方式。使用 Module 的模板：
```
class 网络名字(nn.Module):
    def __init__(self, 一些定义的参数):
        super(网络名字, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)
        self.layer2 = nn.Sequential(...)
        ...

        定义需要用的网络层

    def forward(self, x): # 定义前向传播
        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x = x1 + x2
        ...
        return x
```
- 实现上述神经网络：
```
class module_net(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(module_net, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)
        
        self.layer2 = nn.Tanh()
        
        self.layer3 = nn.Linear(num_hidden, num_output)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


mo_net = module_net(2, 4, 1)
```

- 访问模型中的某层可以直接通过名字
```
# 第一层
l1 = mo_net.layer1
```

- 打印权值
```
print(l1.weight)
```

- 训练模型
```
# 定义优化器
optim = torch.optim.SGD(mo_net.parameters(), 1.)

# 训练1000次
for e in range(10000):
    out = mo_net(Variable(x))
    loss = criterion(out, Variable(y))
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (e + 1) % 1000 == 0:
        print('epoch: {}, loss: {}'.format(e+1, loss.data[0]))

# 保存模型
torch.save(mo_net.state_dict(), 'module_net.pth')

```

- 下载 MNIST 数据
```
def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,)) # 拉平
    x = torch.from_numpy(x)
    return x

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True) # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)
```

- 使用 DataLoader 定义一个数据迭代器
```
from torch.utils.data import DataLoader
train_data = DataLoader(train_set, batch_size=64, shuffle=True)
test_data = DataLoader(test_set, batch_size=128, shuffle=False)

# iter()获取迭代对象的迭代器，next()获取下一条数据
a, a_label = next(iter(train_data))
```

- 绘制图像
```
plt.title('train loss')
plt.plot(np.arange(len(losses)), losses)
```

# 参数初始化
- 可以通过`.weigth`和`.bias`访问网络的权值，并通过`.data`访问其数值，并替换：
```
# 定义一个 Sequential 模型
net1 = nn.Sequential(
    nn.Linear(30, 40),
    nn.ReLU(),
    nn.Linear(40, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

# 定义一个 Tensor 直接对其进行替换
net1[0].weight.data = torch.from_numpy(np.random.uniform(3, 5, size=(40, 30)))

# 模型中相同类型的层需要相同初始化方式
for layer in net1:
    if isinstance(layer, nn.Linear): # 判断是否是线性层
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=param_shape)) 
        # 定义为均值为 0，方差为 0.5 的正态分布
```

- 对于 Module 的参数化，可以直接像 Sequential 一样对其 Tensor 进行重新定义。如果用循环方式，需要介绍两个属性，children 和 modules, children 只访问到模型定义中的第一层，modules 会访问到最后的结构：
```
class sim_net(nn.Module):
    def __init__(self):
        super(sim_net, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU()
        )
        
        self.l1[0].weight.data = torch.randn(40, 30) # 直接对某一层初始化
        
        self.l2 = nn.Sequential(
            nn.Linear(40, 50),
            nn.ReLU()
        )
        
        self.l3 = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.l1(x)
        x =self.l2(x)
        x = self.l3(x)
        return x

net2 = sim_net()

# 访问 children
for i in net2.children():
    print(i)

# 访问 modules
for i in net2.modules():
    print(i)

# 迭代初始化
for layer in net2.modules():
    if isinstance(layer, nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=param_shape))
```

- torch.nn.init
```
from torch.nn import init

init.xavier_uniform(net1[0].weight)
```

# 优化算法
- 随机梯度下降
```
def sgd_update(parameters, lr):
    for param in parameters:
        param.data = param.data - lr * param.grad.data

optimzier = torch.optim.SGD(net.parameters(), 1e-2)
```

- 动量法
```
def sgd_momentum(parameters, vs, lr, gamma):
    for param, v in zip(parameters, vs):
        v[:] = gamma * v + lr * param.grad.data
        param.data = param.data - v

optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.9) # 加动量
```

- Adagrad
```
def sgd_adagrad(parameters, sqrs, lr):
    eps = 1e-10
    for param, sqr in zip(parameters, sqrs):
        sqr[:] = sqr + param.grad.data ** 2
        div = lr / torch.sqrt(sqr + eps) * param.grad.data
        param.data = param.data - div

optimizer = torch.optim.Adagrad(net.parameters(), lr=1e-2)
```

- RMSProp
```
def rmsprop(parameters, sqrs, lr, alpha):
    eps = 1e-10
    for param, sqr in zip(parameters, sqrs):
        sqr[:] = alpha * sqr + (1 - alpha) * param.grad.data ** 2
        div = lr / torch.sqrt(sqr + eps) * param.grad.data
        param.data = param.data - div

optimizer = torch.optim.RMSprop(net.parameters(), lr=1e-3, alpha=0.9)
```

- Adadelta
```
def adadelta(parameters, sqrs, deltas, rho):
    eps = 1e-6
    for param, sqr, delta in zip(parameters, sqrs, deltas):
        sqr[:] = rho * sqr + (1 - rho) * param.grad.data ** 2
        cur_delta = torch.sqrt(delta + eps) / torch.sqrt(sqr + eps) * param.grad.data
        delta[:] = rho * delta + (1 - rho) * cur_delta ** 2
        param.data = param.data - cur_delta

optimizer = torch.optim.Adadelta(net.parameters(), rho=0.9)
```

- Adam
```
def adam(parameters, vs, sqrs, lr, t, beta1=0.9, beta2=0.999):
    eps = 1e-8
    for param, v, sqr in zip(parameters, vs, sqrs):
        v[:] = beta1 * v + (1 - beta1) * param.grad.data
        sqr[:] = beta2 * sqr + (1 - beta2) * param.grad.data ** 2
        v_hat = v / (1 - beta1 ** t)
        s_hat = sqr / (1 - beta2 ** t)
        param.data = param.data - lr * v_hat / torch.sqrt(s_hat + eps)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
```

