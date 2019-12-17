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