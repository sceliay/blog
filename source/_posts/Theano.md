---
categories: Machine Learning
title: Theano
date: 2018-12-19 18:43:23
tags: [Theano, Machine Learning, Deep Learning]
---

准备用Theano做一些实验，记录一下Theano的语法。参考：[Theano教程系列]

1. 加载theano和numpy模块，并创建`function`:
```
import numpy as np
import theano.tensor as T
from theano import function
```

2. 定义常量(scalar)及函数(function)：
```
x = T.dscalar('x')
y = T.dscalar('y')
z = x+y

f = function([x,y],z)

print(f(2,3))
# 5.0
```

3. 打印原函数
```
from theano import pp

print(pp(z))
# (x+y)
```

4. 定义矩阵
```
x = T.dmatrix('x')
y = T.dmatrix('y')
z = x+y

printf(
	f(np.arange(12).reshape(3,4),10*np.ones(3,4))
)

'''
[[ 10.  11.  12.  13.]
 [ 14.  15.  16.  17.]
 [ 18.  19.  20.  21.]]
 '''
```

5. Function的用法
```
import numpy as np
import theano.tensor as T
import theano
```

- 激活函数
```
x = T.dmatrix('x')
s = 1/(1+T.exp(-x))

logistic = theano.function([x],s)

print(logistic([[0,1],[-2,-3]]))

'''
[[ 0.5         0.73105858]
 [ 0.26894142  0.11920292]]
```

