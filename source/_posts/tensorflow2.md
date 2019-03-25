---
title: Tensorflow学习笔记2
date: 2019-03-25 19:29:23
tags: [Tensorflow, Python, Machine Learning]
---
参考资料：[Keras中文文档](https://keras.io/zh/), [W3CSCHOOL](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-bm7y28si.html)

1. [namedtuple](https://www.cnblogs.com/herbert/p/3468294.html)
namedtuple创建一个和tuple类似的对象，而且对象拥有可以访问的属性。这对象更像带有数据属性的类，不过数据属性是只读的。
eg: `TPoint = namedtuple('TPoint', ['x', 'y'])` 创建一个TPoint类型，而且带有属性x, y.

2. [isinstance](http://www.runoob.com/python/python-func-isinstance.html)
用来判断一个对象是否是一个已知的类型，类似 type()。
`isinstance()` 与 `type()` 区别：
- `type()` 不会认为子类是一种父类类型，不考虑继承关系。
- `isinstance()` 会认为子类是一种父类类型，考虑继承关系。
使用：`isinstance(object, classinfo)`

3. [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)
- [Input](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Input): 定义模型的输入
- [Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding): 定义嵌入层[参考](http://frankchen.xyz/2017/12/18/How-to-Use-Word-Embedding-Layers-for-Deep-Learning-with-Keras/)
  - Keras提供了一个嵌入层，适用于文本数据的神经网络。
  - 它要求输入数据是整数编码的，所以每个字都用一个唯一的整数表示。这个数据准备步骤可以使用Keras提供的Tokenizer API来执行。
  - 嵌入层用随机权重进行初始化，并将学习训练数据集中所有单词的嵌入。
  - `e = Embedding(input_dim=200, output_dim=32, input_length=50)`	
- [add](https://www.tensorflow.org/api_docs/python/tf/keras/layers/add): 将两个输出加和
- [Concatenate]


