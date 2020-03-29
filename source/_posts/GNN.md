---
categories: Machine Learning
title: GNN
date: 2020-03-01 14:57:50
tags: [Machine Learning]
---

参考： [DGL](https://docs.dgl.ai/tutorials/models/index.html#revisit-classic-models-from-a-graph-perspective), [Transformers are Graph Neural Networks](https://graphdeeplearning.github.io/post/transformers-are-gnns/)

# [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/)

1. The hidden layer can be used as a lookup table.
2. Perceptron: 感知机
3. Word Embedding: a paramaterized function mapping words in some language to high-dimensional vectors (perhaps 200 to 500 dimensions).
4. Visualizing: t-SNE
5. Shared Representations:
	- Learning a good representation on a task A and then using it on a task B.
		- pretraining, transfer learning, and multi-task learning.
	- Map multiple kinds of data into a single representation.
		- bilingual word-embedding, 
		- embed images and words in a single representation.

# [Attention](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
1. Born for Translation:
	- encoder hidden states (bidirectional RNN)
	- decoder hidden states 
	- alignment between source and target (alignment score)
2. Attention Mechanisms
	- self-attention
	- soft vs. hard attention
	- global vs. local attention
3. Neural Turing Machines
4. Pointer Network
5. Transformer
6. Simple Neural Attention Meta-Learner (SNAIL)
7. Self-Attention GAN

