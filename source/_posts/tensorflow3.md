---
categories: Machine Learning
title: Tensorflow 小记
date: 2019-03-25 19:29:23
tags: [Tensorflow, Python, Machine Learning]
---

1. 输出tensor值
```
import tensorflow as tf
x = tf.constant(1)
with tf.Session() as sess:
    print sess.run(x)
```

2. (矩阵加和)[https://blog.csdn.net/qq_16137569/article/details/72568793]
`tf.reduce_sum(input_tenosr, reduction_indices, keep_dims, name)`
```
x = [[1,1,1], [1,1,1]]

tf.reduce_sum(x) # 6
tf.reduce_sum(x,0) # [2,2,2]
tf.reduce_sum(x,1) # [3,3]
tf.reduce_sum(x,1, keep_dims=True) # [[3],[3]]
tf.reduce_sum(x,[0,1]) # 6
```

3. (获取中间某一层的输出)[https://www.cnblogs.com/as3asddd/p/10129241.html]
```
model.fit(data, labels, epochs=10, batch_size=32)

# 已有的model在load权重过后
# 取某一层的输出为输出新建为model，采用函数模型
dense1_layer_model = Model(inputs=model.input,
	outputs=model.get_layer('Dense_1').output)

# layer_model = Model(inputs=model.input, outputs=model.layers[6].output)

#以这个model的预测值作为输出
dense1_output = dense1_layer_model.predict(data)
```

4. (矩阵上下三角)[https://blog.csdn.net/ACM_hades/article/details/88790013]
- `tf.matrix_band_part`
- 2.0版本： `tf.linalg.band_par`
- ```
tf.linalg.band_part(
    input,
    num_lower,
    num_upper,
    name=None
)
```
作用：主要功能是以对角线为中心，取它的副对角线部分，其他部分用0填充。
input:输入的张量.
num_lower:下三角矩阵保留的副对角线数量，从主对角线开始计算，相当于下三角的带宽。取值为负数时，则全部保留。
num_upper:上三角矩阵保留的副对角线数量，从主对角线开始计算，相当于上三角的带宽。取值为负数时，则全部保留。

5. 张量扩充
`tf.tile(raw, multiples=[2, 1])`