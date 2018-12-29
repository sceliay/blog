---
categories: Machine Learning
title: Theano
date: 2018-12-19 18:43:23
tags: [Theano, Machine Learning, Deep Learning]
---


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

- 多输入/输出
```
a,b = T.dmatrices('a','b')

diff = a-b
abs_diff = abs(a-b)

f = theano.function([a,b],[diff,abs_diff])

x1,x2= f(
    np.ones((2,2)), # a
    np.arange(4).reshape((2,2))  # b
)

"""
array([[ 1.,  0.],
       [-1., -2.]]),
array([[ 1.,  0.],
       [ 1.,  2.]]),  
"""
```

- 默认值 & 名字
```
x,y,w = T.dscalars('x','y','w')
z = (x+y)*w

f = theano.function([x,
                     theano.In(y,value=1),
                     theano.In(w,value=2)],
                    z)

print(f(23))    # 使用默认
print(f(23,1,4)) # 不使用默认
"""
48.0
100.0
"""

f = theano.function([x,
                     theano.In(y,value=1),
                     theano.In(w,value=2,name='weights')],
                    z)
                    
print (f(23,1,weights=4)) ##调用方式

"""
100.0
"""
```

- Shared变量
```
state = theano.shared(np.array(0,dtype=np.float64), 'state')     # inital state = 0
inc = T.scalar('inc', dtype=state.dtype)
accumulator = theano.function([inc], state, updates=[(state, state+inc)]) # updates: state+=inc

# to get variable value
print(state.get_value())
# 0.0

accumulator(1)   # return previous value, 0 in here
print(state.get_value())
# 1.0

accumulator(10)  # return previous value, 1 in here
print(state.get_value())
# 11.0

state.set_value(-1)
accumulator(3)
print(state.get_value())
# 2.0
```

- [激活函数](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html)：
`theano.tensor.nnet.nnet.sigmoid(x)`
`softplus()`, `relu()`, `softmax()`,`tanh()`

- Layer类
```
l1 = Layer(inputs, in_size, out_size, T.nnet.relu)
l2 = Layer(l1.outputs, in_size, out_size, None) # None采用默认的线性激活函数

class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation_function=None):
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
        self.b = theano.shared(np.zeros((out_size, )) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)
```



