---
categories: Machine Learning
title: Pytorch 时序
date: 2019-12-17 19:02:52
tags: [Machine Learning, Python, Pytorch]
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

参考：[code of learn deep learning with pytroch](https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch/blob/master/chapter2_PyTorch-Basics/PyTorch-introduction.ipynb)

# 循环神经网络
## RNN
$$ h_t = tanh(w_{ih}\*x_t+b_{ih}+w_{hh}\*h_{t-1}+b_{hh}) $$	

1. `torch.nn.RNNCell()` 只接受序列中单步的输入，且必须传入隐藏状态
	- `input_size`: 输入的特征维度
	- `hidden_size`: 输出的特征维度
	- `num_layers`: 网络的层数
	- `nonlinearity`: 非线性激活函数，默认 'tanh'
	- `bias`: 是否使用偏置，默认使用
	- `batch_first`: 输入数据的形式，默认 False，为(seq, batch, feature)，
	- `dropout`: 是否应用 dropout
	- `bidirectional`: 是否使用双向 rnn，默认 False
```
import torch
from torch.autograd import Variable
from torch import nn

# 定义一个单步的 rnn
rnn_single = nn.RNNCell(input_size=100, hidden_size=200)

# 访问其中的参数
rnn_single.weight_hh

# 构造一个序列，长为 6，batch 是 5， 特征是 100
x = Variable(torch.randn(6, 5, 100)) # 这是 rnn 的输入格式

# 定义初始的记忆状态
h_t = Variable(torch.zeros(5, 200))

# 传入 rnn
out = []
for i in range(6): # 通过循环 6 次作用在整个序列上
    h_t = rnn_single(x[i], h_t)
    out.append(h_t)
```
2. `torch.nn.RNN()` 可以接受一个序列的输入，默认为全0的隐藏状态，可以自己申明
	- `input_size`
	- `hidden_size`
	- `bias`
	- `nonlinearity`
```
rnn_seq = nn.RNN(100, 200)

# 访问其中的参数
rnn_seq.weight_hh_l0

# 使用默认的全 0 隐藏状态
out, h_t = rnn_seq(x) 

# 自己定义初始的隐藏状态
h_0 = Variable(torch.randn(1, 5, 200))
out, h_t = rnn_seq(x, h_0)
```
输出的结果均为 (seq, batch, feature)

## LSTM
$$ f_t = \sigma (W_f \cdot [h_{t-1},x_t]+b_f) $$
$$ i_t = \sigma (W_i \cdot [h_{t-1},x_t]+b_i) $$
$$ \tilde{C_t} = tanh (W_C \cdot [h_{t-1},x_t]+b_C) $$
$$ C_t = f_t\*C_{t-1}+i_t \*\tilde{C_t} $$
$$ o_t = \sigma(w_o \cdot [h_{t-1},x_t]+b_o) $$
$$ h_t = o_t\*tanh(C_t) $$
LSTM 与基本 RNN 一样，参数也相同，具有`nn.LSTMCell()`和`nn.LSTM()`两种形式。
```
lstm_seq = nn.LSTM(50, 100, num_layers=2) # 输入维度 50，输出 100，两层

lstm_seq.weight_hh_l0 # 第一层的 h_t 权重

lstm_input = Variable(torch.randn(10, 3, 50)) # 序列 10，batch 是 3，输入维度 50

out, (h, c) = lstm_seq(lstm_input) # 使用默认的全 0 隐藏状态

# 不使用默认的隐藏状态
h_init = Variable(torch.randn(2, 3, 100))
c_init = Variable(torch.randn(2, 3, 100))

out, (h, c) = lstm_seq(lstm_input, (h_init, c_init))
```

## GRU
$$ z_t = \sigma (W_z \cdot [h_{t-1},x_t]) $$
$$ r_t = \sigma (W_r \cdot [h_{t-1},x_t]) $$
$$ \tilde{h_t} = tanh(W \cdot [r_t \* h_{t-1}, x_t]) $$
$$ h_t = (1-z_t)\*h_{t-1}+z_t \* \tilde{h_t} $$
```
gru_seq = nn.GRU(10, 20)
gru_input = Variable(torch.randn(3, 32, 10))

out, h = gru_seq(gru_input)
```

## RNN用于时间序列分析
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable

data_csv = pd.read_csv('./data.csv', usecols=[1])

# 数据预处理
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: (x-min_value) / scalar, dataset))

# 通过前两个月的流量来预测当月流量
def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# 创建好输入输出
data_X, data_Y = create_dataset(dataset)

# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

# RNN 读入数据维度为 (seq,batch,feature)
train_X = train_X.reshape(-1, 1, 2)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 2)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)

# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
        
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers) # rnn
        self.reg = nn.Linear(hidden_size, output_size) # 回归
        
    def forward(self, x):
        x, _ = self.rnn(x) # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s*b, h) # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x
# 输入维度为2，隐层为4
net = lstm_reg(2, 4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

# 开始训练
for e in range(1000):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0: # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.data[0]))

# 预测结果
net = net.eval() # 转换成测试模式

data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = net(var_data) # 测试集的预测结果

# 改变输出的格式，view=reshape
pred_test = pred_test.view(-1).data.numpy()

# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')
```
# Embedding
```
import torch
from torch import nn
from torch.autograd import Variable

# 定义词嵌入
embeds = nn.Embedding(2, 5) # 2 个单词，维度 5

# 得到词嵌入矩阵
embeds.weight

# 直接手动修改词嵌入的值
embeds.weight.data = torch.ones(2, 5)

# 访问第 50 个词的词向量
embeds = nn.Embedding(100, 10)
single_word_embed = embeds(Variable(torch.LongTensor([50])))
```

# N-Gram
```
CONTEXT_SIZE = 2 # 依据的单词数
EMBEDDING_DIM = 10 # 词向量的维度
# 我们使用莎士比亚的诗
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# 创建数据集
trigram = [((test_sentence[i], test_sentence[i+1]), test_sentence[i+2]) 
            for i in range(len(test_sentence)-2)]

# 建立每个词与数字的编码，据此构建词嵌入
vocb = set(test_sentence) # 使用 set 将重复的元素去掉
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

# 定义模型
class n_gram(nn.Module):
    def __init__(self, vocab_size, context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM):
        super(n_gram, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, vocab_size)
        )
        
    def forward(self, x):
        voc_embed = self.embed(x) # 得到词嵌入
        voc_embed = voc_embed.view(1, -1) # 将两个词向量拼在一起
        out = self.classify(voc_embed)
        return out

net = n_gram(len(word_to_idx))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-5)

for e in range(100):
    train_loss = 0
    for word, label in trigram: # 使用前 100 个作为训练集
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word])) # 将两个词作为输入
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # 前向传播
        out = net(word)
        loss = criterion(out, label)
        train_loss += loss.data
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 20 == 0:
        print('epoch: {}, Loss: {:.6f}'.format(e + 1, train_loss / len(trigram)))

net = net.eval()

# 测试一下结果
word, label = trigram[19]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = net(word)

pred_label_idx = out.max(1)[1].data.numpy()[0]
print(pred_label_idx)
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}'.format(label, predict_word))
```

# LSTM 做词性预测
```
import torch
from torch import nn
from torch.autograd import Variable

training_data = [("The dog ate the apple".split(),
                  ["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(), 
                  ["NN", "V", "DET", "NN"])]

# 对单词和标签编码
word_to_idx = {}
tag_to_idx = {}
for context, tag in training_data:
    for word in context:
        if word.lower() not in word_to_idx:
            word_to_idx[word.lower()] = len(word_to_idx)
    for label in tag:
        if label.lower() not in tag_to_idx:
            tag_to_idx[label.lower()] = len(tag_to_idx)

# 对字母编码
alphabet = 'abcdefghijklmnopqrstuvwxyz'
char_to_idx = {}
for i in range(len(alphabet)):
    char_to_idx[alphabet[i]] = i

# 构建训练数据
def make_sequence(x, dic): # 字符编码
    idx = [dic[i.lower()] for i in x]
    idx = torch.LongTensor(idx)
    return idx

make_sequence('apple', char_to_idx)

# 构建单个字符的 lstm 模型
class char_lstm(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(char_lstm, self).__init__()
        
        self.char_embed = nn.Embedding(n_char, char_dim)
        self.lstm = nn.LSTM(char_dim, char_hidden)
        
    def forward(self, x):
        x = self.char_embed(x)
        out, _ = self.lstm(x)
        return out[-1] # (batch, hidden)

# 构建词性分类的 lstm 模型
class lstm_tagger(nn.Module):
    def __init__(self, n_word, n_char, char_dim, word_dim, 
                 char_hidden, word_hidden, n_tag):
        super(lstm_tagger, self).__init__()
        self.word_embed = nn.Embedding(n_word, word_dim)
        self.char_lstm = char_lstm(n_char, char_dim, char_hidden)
        self.word_lstm = nn.LSTM(word_dim + char_hidden, word_hidden)
        self.classify = nn.Linear(word_hidden, n_tag)
        
    def forward(self, x, word):
        char = []
        for w in word: # 对于每个单词做字符的 lstm
            char_list = make_sequence(w, char_to_idx)
            char_list = char_list.unsqueeze(1) # (seq, batch, feature) 满足 lstm 输入条件
            char_infor = self.char_lstm(Variable(char_list)) # (batch, char_hidden)
            char.append(char_infor)
        char = torch.stack(char, dim=0) # (seq, batch, feature)
        
        x = self.word_embed(x) # (batch, seq, word_dim)
        x = x.permute(1, 0, 2) # 改变顺序
        x = torch.cat((x, char), dim=2) # 沿着特征通道将每个词的词嵌入和字符 lstm 输出的结果拼接在一起
        x, _ = self.word_lstm(x)
        
        s, b, h = x.shape
        x = x.view(-1, h) # 重新 reshape 进行分类线性层
        out = self.classify(x)
        return out

net = lstm_tagger(len(word_to_idx), len(char_to_idx), 10, 100, 50, 128, len(tag_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

# 开始训练
for e in range(300):
    train_loss = 0
    for word, tag in training_data:
        word_list = make_sequence(word, word_to_idx).unsqueeze(0) # 添加第一维 batch
        tag = make_sequence(tag, tag_to_idx)
        word_list = Variable(word_list)
        tag = Variable(tag)
        # 前向传播
        out = net(word_list, word)
        loss = criterion(out, tag)
        train_loss += loss.data[0]
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 50 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, train_loss / len(training_data)))

# 预测
net = net.eval()

test_sent = 'Everybody ate the apple'
test = make_sequence(test_sent.split(), word_to_idx).unsqueeze(0)
out = net(Variable(test), test_sent.split())

print(tag_to_idx)
```