---
categories: Machine Learning
title: Attention
date: 2019-12-04 14:53:53
tags: [Machine Learning]
---
# Attention
参考：[Attn: Illustrated Attention](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)

随着深度学习的发展，Neural Machine Translation(NMT) 也逐渐取代传统的 Statistical Machine Translation(SMT)。 其中，最广为人知的框架为 [Sutskever et. al](https://arxiv.org/abs/1409.3215) 提出的 sequence-to-sequence(seq2seq) 模型。

Seq2seq 是一个含有两个 RNN 的 encoder-decoder 模型： encoder，按字符读入从而获得一个固定长度的表示； decoder,根据这些输入训练另一个 RNN 从而获得顺序输出。

这样的模型会导致最终 decoder 获得的是 encoder 最后输出的 hidden state，当文本过长时，容易遗忘开始输入的字段。

Attention 可以作为 encoder 与 decoder 的中间接口，为 decoder 提供每一个 encoder hidden state 的信息。如此，模型能够选择性的关注有用的部分，并学习 encoder 与 decoder 中字句的对齐。

Attention 有两种类型： global attention 使用所有的 encoder hidden state，local attention 只使用部分。 Attention layer 的实现可分为4个步骤。

0. 准备 hidden states.
例子中包含4个 encoder hidden states 和 current decoder hidden state.
Note: 最后一个 encoder hidden state 作为 decoder 的第一个 time step 输入。 第一个 time step 的输出被称为第一个 decoder hidden state.

1. 计算每个 encoder hidden state 的 score.
可以通过一个 score function（也被称为 alignment score function 或 alignment model）来计算 score。在本例中，score function 为 dot product。更多 [score function](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3#ba24)。
计算公式：`score = decoder hidden state x encoder hidden state`
```
decoder_hidden = [10, 5, 10]
encoder_hidden  score
---------------------
     [0, 1, 1]     15 (= 10×0 + 5×1 + 10×1, the dot product)
     [5, 0, 1]     60
     [1, 1, 0]     15
     [0, 5, 1]     35
```

2. 将所有 scores 通过一个 softmax 层。
通过 softmax 层，将所有 softmaxed scores 加和为1，被称为 attention distribution.
```
encoder_hidden  score  score^
-----------------------------
     [0, 1, 1]     15       0
     [5, 0, 1]     60       1
     [1, 1, 0]     15       0
     [0, 5, 1]     35       0
```

3. 将每个 encoder hidden state 乘以对应 softmaxed score.
通过 encoder hidden state 乘以相应 softmaxed score，可以获得 alignment vector 或 annotation vector。
```
encoder  score  score^  alignment
---------------------------------
[0, 1, 1]   15      0   [0, 0, 0]
[5, 0, 1]   60      1   [5, 0, 1]
[1, 1, 0]   15      0   [0, 0, 0]
[0, 5, 1]   35      0   [0, 0, 0]
```
这里，表示第一个翻译的词与嵌入`[5,0,1]`的输入相对应。

4. 将 alignment vectors 相加。
将 alignment vectors 相加获得 context vector。

5. 将 context vector 输入 decoder.


# Self-attention
参考：[Illustrated: Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
1. 输入
例子中，输入为3个4维向量：
```
Input 1: [1, 0, 1, 0] 
Input 2: [0, 2, 0, 2]
Input 3: [1, 1, 1, 1]
```

2. 初始化权值
每个输入有三个表示：key, query, value。例子中，这些表示用3维向量表示，则权值为4\*3矩阵。
Note: value的维度与输出相同。
- key 的权值
```
[[0, 0, 1],
 [1, 1, 0],
 [0, 1, 0],
 [1, 1, 0]]
```
- query 的权值
```
[[1, 0, 1],
 [1, 0, 0],
 [0, 0, 1],
 [0, 1, 1]]
```
- value 的权值
```
[[0, 2, 0],
 [0, 3, 0],
 [1, 0, 3],
 [1, 1, 0]]
```
Nots: 在神经网络中，权值常由合适的随机分布来初始化，如 Gaussian, Xavier 和 Kaiming 分布。

3. 计算 key, query 和 value
计算公式为：`input x weight`
- key:
```
               [0, 0, 1]
[1, 0, 1, 0]   [1, 1, 0]   [0, 1, 1]
[0, 2, 0, 2] x [0, 1, 0] = [4, 4, 0]
[1, 1, 1, 1]   [1, 1, 0]   [2, 3, 1]
```
- query:
```
               [1, 0, 1]
[1, 0, 1, 0]   [1, 0, 0]   [1, 0, 2]
[0, 2, 0, 2] x [0, 0, 1] = [2, 2, 2]
[1, 1, 1, 1]   [0, 1, 1]   [2, 1, 3]
```
- value:
```
               [0, 2, 0]
[1, 0, 1, 0]   [0, 3, 0]   [1, 2, 3] 
[0, 2, 0, 2] x [1, 0, 3] = [2, 8, 0]
[1, 1, 1, 1]   [1, 1, 0]   [2, 6, 3]
```
Note: 有时候也可以加上偏置。

4. 计算 Input 1 的 attention score
计算公式为：` Input 1's query x keys^T`
```
            [0, 4, 2]
[1, 0, 2] x [1, 4, 3] = [2, 4, 4]
            [1, 0, 1]
```
Note: 上述为 dot product attention, 其他 [score function](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3) 有 scaled dot product 和 additive/concat。

5. 计算softmax
对 attention score 进行 softmax 计算:
```
softmax([2, 4, 4]) = [0.0, 0.5, 0.5]
```

6. 将 score 乘以 value:
将 attention score 乘以对应的 value, 获得 weighted values:
```
1: 0.0 * [1, 2, 3] = [0.0, 0.0, 0.0]
2: 0.5 * [2, 8, 0] = [1.0, 4.0, 0.0]
3: 0.5 * [2, 6, 3] = [1.0, 3.0, 1.5]
```

7. 将 weighted values 加和获得 Output 1:
```
  [0.0, 0.0, 0.0]
+ [1.0, 4.0, 0.0]
+ [1.0, 3.0, 1.5]
-----------------
= [2.0, 7.0, 1.5]
```

8. 对于 Input 2 和 Input 3 重复4-7操作
Note: query 和 key 的维度需要保持一致，而 value 的维度与 output 一致。