---
categories: Machine Learning
title: Keras学习笔记
date: 2019-03-25 19:29:23
tags: [Keras, Tensorflow, Python, Machine Learning]
---
参考资料：[Keras中文文档](https://keras.io/zh/),[Keras英文文档](https://keras.io/), [Keras教程](https://blog.csdn.net/u014061630/article/details/81086564), [W3CSCHOOL](https://www.w3cschool.cn/tensorflow_python/tensorflow_python-5dym2s6j.html),[Tensorflow中文社区](https://www.tensorflow.org/api_docs/python/tf/keras),

1. `Sequential`顺序模型
- 定义：
```
from keras.models import Sequential
model = Sequential()
```
- 使用`add`堆叠模型：
```
from keras.layers import Dense
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```
  - `activation`: 激活函数
  - `kernel_initializer`和`bias_initializer`: 层创建时，权值和偏差的初始化方法，默认为`Glorot uniform`
  - `kernel_regularizer`和`bias_regularizer`：层的权重、偏差的正则化方法。
  ```
  # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
  layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
  # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
  layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))

  # A linear layer with a kernel initialized to a random orthogonal matrix:
  layers.Dense(64, kernel_initializer='orthogonal')
  # A linear layer with a bias vector initialized to 2.0s:
  layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))
  ```

- 使用`compile`配置学习过程
```
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

- 配置优化器
```
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

- 训练数据
```
# x_train 和 y_train 是 Numpy 数组 -- 就像在 Scikit-Learn API 中一样。
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

- 评估模型
```
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

- 预测
```
classes = model.predict(x_test, batch_size=128)
```

- [更多实例](https://keras.io/getting-started/sequential-model-guide/)

2. `Functional API`函数式API
- 可用于定义更复杂的模型。
- 层可调用，返回值为一个tensor
- 输入tensors和输出tensors被用来定义一个`tf.keras.model`实例
- 训练方法与sequential一样
- eg.
```
inputs = keras.Input(shape=(32,))  # Returns a placeholder tensor

# A layer instance is callable on a tensor, and returns a tensor.
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

# Instantiate the model given inputs and outputs.
model = keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)
```
- [更多实例](https://keras.io/zh/getting-started/functional-api-guide/)

3. 模型的方法和属性
- `model.layers` 是包含模型网络层的展平列表。
- `model.inputs` 是模型输入张量的列表。
- `model.outputs` 是模型输出张量的列表。
- `model.summary()` 打印出模型概述信息。 它是 utils.print_summary 的简捷调用。
- `model.get_config()` 返回包含模型配置信息的字典。通过以下代码，就可以根据这些配置信息重新实例化模型：
- `model.get_weights()` 返回模型中所有权重张量的列表，类型为 Numpy 数组。
- `model.set_weights(weights)` 从 Numpy 数组中为模型设置权重。列表中的数组必须与`get_weights()`返回的权重具有相同的尺寸。
- `model.save_weights(filepath)` 将模型权重存储为 HDF5 文件。
- `model.load_weights(filepath, by_name=False)`: 从 HDF5 文件（由 save_weights 创建）中加载权重。默认情况下，模型的结构应该是不变的。 如果想将权重载入不同的模型（部分层相同）， 设置 by_name=True 来载入那些名字相同的层的权重。

4. Model类继承
通过编写`tf.keras.Model`的子类来构建一个自定义模型。在`init`方法里创建 layers。在`call`方法里定义前向传播过程。在`call`中，你可以指定自定义的损失函数，通过调用`self.add_loss(loss_tensor)`。
```
import keras

class SimpleMLP(keras.Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.dense1 = keras.layers.Dense(32, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        x = self.dense1(inputs)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)

model = SimpleMLP()
model.compile(...)
model.fit(...)
```
在类继承模型中，模型的拓扑结构是由 Python 代码定义的（而不是网络层的静态图）。这意味着该模型的拓扑结构不能被检查或序列化。因此，以下方法和属性不适用于类继承模型：
- `model.inputs` 和 `model.outputs`。
- `model.to_yaml()` 和 `model.to_json()`。
- `model.get_config()` 和 `model.save()`。

5. [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)
- [Input](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Input): 定义模型的输入
- [Embedding](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding): 定义嵌入层[[参考](http://frankchen.xyz/2017/12/18/How-to-Use-Word-Embedding-Layers-for-Deep-Learning-with-Keras/)]
  - Keras提供了一个嵌入层，适用于文本数据的神经网络。
  - 它要求输入数据是整数编码的，所以每个字都用一个唯一的整数表示。这个数据准备步骤可以使用Keras提供的Tokenizer API来执行。
  - 嵌入层用随机权重进行初始化，并将学习训练数据集中所有单词的嵌入。
  - `e = Embedding(input_dim=200, output_dim=32, input_length=50)`	
- [add](https://www.tensorflow.org/api_docs/python/tf/keras/layers/add): 将两个输出加和
- [Concatenate](https://www.tensorflow.org/api_docs/python/tf/keras/layers/concatenate): 链接两个张量
- [dot](https://www.programcreek.com/python/example/89694/keras.layers.dot)
 

