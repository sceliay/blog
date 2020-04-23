---
categories: Machine Learning
title: tensorflow-2.x
date: 2020-03-30 16:08:29
tags: [Tensorflow, Python, Machine Learning]
---

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

Tensorflow 2.x的学习笔记。主要教程为[Deep-Learning-with-TensorFlow-book](https://github.com/dragen1860/Deep-Learning-with-TensorFlow-book)

# Tensorflow 2.x vs. 1.x
- Tensorflow 1.x 先创建计算图，然后运行，为静态图模式，该编程方法叫做符号式编程。
```
import tensorflow as tf
# 1.创建计算图阶段
# 创建2个输入端子，指定类型和名字
a_ph = tf.placeholder(tf.float32, name='variable_a')
b_ph = tf.placeholder(tf.float32, name='variable_b')
# 创建输出端子的运算操作，并命名
c_op = tf.add(a_ph, b_ph, name='variable_c')

# 2.运行计算图阶段
# 创建运行环境
sess = tf.InteractiveSession()
# 初始化步骤也需要作为操作运行
init = tf.global_variables_initializer()
sess.run(init) # 运行初始化操作，完成初始化
# 运行输出端子，需要给输入端子赋值
c_numpy = sess.run(c_op, feed_dict={a_ph: 2., b_ph: 4.})
# 运算完输出端子才能得到数值类型的c_numpy
print('a+b=',c_numpy)

```
Tensorflow 2.x 支持动态图优先模式，即为命令式编程。
```
import tensorflow as tf
# 1.创建输入张量
a = tf.constant(2.)
b = tf.constant(4.)
# 2.直接计算并打印
print('a+b=',a+b)
```

# 核心功能
- 加速计算（GPU vs. CPU)
```
# 创建在CPU 上运算的2 个矩阵
with tf.device('/cpu:0'):
	cpu_a = tf.random.normal([1, n])
	cpu_b = tf.random.normal([n, 1])
	print(cpu_a.device, cpu_b.device)
# 创建使用GPU 运算的2 个矩阵
with tf.device('/gpu:0'):
	gpu_a = tf.random.normal([1, n])
	gpu_b = tf.random.normal([n, 1])
	print(gpu_a.device, gpu_b.device)

def cpu_run():
	with tf.device('/cpu:0'):
		c = tf.matmul(cpu_a, cpu_b)
	return c
def gpu_run():
	with tf.device('/gpu:0'):
		c = tf.matmul(gpu_a, gpu_b)
	return c

# 第一次计算需要热身，避免将初始化阶段时间结算在内
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('warmup:', cpu_time, gpu_time)
# 正式计算10 次，取平均时间
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('run time:', cpu_time, gpu_time)
```
当 $$n>10^4$$ 后，CPU计算速度明显上升。

- 自动梯度
```
import tensorflow as tf
# 创建4 个张量
a = tf.constant(1.)
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)
with tf.GradientTape() as tape:# 构建梯度环境
	tape.watch([w]) # 将w 加入梯度跟踪列表
	# 构建计算过程
	y = a * w**2 + b * w + c
# 求导
[dy_dw] = tape.gradient(y, [w])
print(dy_dw) # 打印出导数
# tf.Tensor(10.0, shape=(), dtype=float32)
```

- 常用神经网络接口

# 回归问题
- 神经元线性模型：
$$ y=wx+b $$
- 平方和误差：
$$ L = \frac{1}{n} \sum_{i=1}^{n}(wx^{(i)}+b-y^{(i)})^2 $$
- 最优参数：
$$ w^*, b^* = \mathop{\arg\min}_{w,b} \frac{1}{n} \sum_{i=1}^{n}(wx^{(i)}+b-y^{(i)})^2 $$
- 梯度下降法：迭代更新
$$ w' = w - \eta \frac{\partial L}{\partial w} $$
$$ b' = w- \eta \frac{\partial L}{\partial b} $$
- 实例：
	- 采集数据： $$ y=1.477x+0.089+\epsilon, \epsilon \sim \mathbb{N}(0,0.01) $$
	```
	data = []# 保存样本集的列表
	for i in range(100): # 循环采样100 个点
		x = np.random.uniform(-10., 10.) # 随机采样输入x
		# 采样高斯噪声
		eps = np.random.normal(0., 0.1)
		# 得到模型的输出
		y = 1.477 * x + 0.089 + eps
		data.append([x, y]) # 保存样本点
	data = np.array(data) # 转换为2D Numpy 数组
	```
	- 计算误差
	```
	def mse(b, w, points):
		# 根据当前的w,b 参数计算均方差损失
		totalError = 0
		for i in range(0, len(points)): # 循环迭代所有点
			x = points[i, 0] # 获得i 号点的输入x
			y = points[i, 1] # 获得i 号点的输出y
			# 计算差的平方，并累加
			totalError += (y - (w * x + b)) ** 2
		# 将累加的误差求平均，得到均方差
		return totalError / float(len(points))
	```
	- 计算梯度
	```
	def step_gradient(b_current, w_current, points, lr):
		# 计算误差函数在所有点上的导数，并更新w,b
		b_gradient = 0
		w_gradient = 0
		M = float(len(points)) # 总样本数
		for i in range(0, len(points)):
			x = points[i, 0]
			y = points[i, 1]
			# 误差函数对b 的导数：grad_b = 2(wx+b-y)，参考公式(2.3)
			b_gradient += (2/M) * ((w_current * x + b_current) - y)
			# 误差函数对w 的导数：grad_w = 2(wx+b-y)*x，参考公式(2.2)
			w_gradient += (2/M) * x * ((w_current * x + b_current) - y)
		# 根据梯度下降算法更新 w',b',其中lr 为学习率
		new_b = b_current - (lr * b_gradient)
		new_w = w_current - (lr * w_gradient)
		return [new_b, new_w]
	```
	- 梯度更新
	```
	def gradient_descent(points, starting_b, starting_w, lr, num_iterations):
		# 循环更新w,b 多次
		b = starting_b # b 的初始值
		w = starting_w # w 的初始值
		# 根据梯度下降算法更新多次
		for step in range(num_iterations):
			# 计算梯度并更新一次
			b, w = step_gradient(b, w, np.array(points), lr)
			loss = mse(b, w, points) # 计算当前的均方差，用于监控训练进度
			if step%50 == 0: # 打印误差和实时的w,b 值
				print(f"iteration:{step}, loss:{loss}, w:{w}, b:{b}")
		return [b, w] # 返回最后一次的w,b
	```
	- 主函数
	```
	def main():
		# 加载训练集数据，这些数据是通过真实模型添加观测误差采样得到的
		lr = 0.01 # 学习率
		initial_b = 0 # 初始化b 为0
		initial_w = 0 # 初始化w 为0
		num_iterations = 1000
		# 训练优化1000 次，返回最优w*,b*和训练Loss 的下降过程
		[b, w], losses = gradient_descent(data, initial_b, initial_w, lr, num_it
		erations)
		loss = mse(b, w, data) # 计算最优数值解w,b 上的均方差
		print(f'Final loss:{loss}, w:{w}, b:{b}')
	```

# 分类问题
- 非线性模型
激活函数： sigmoid, relu
$$ o = \sigma(wx+b) $$
- 下载MNIST数据:
```
import os
import tensorflow as tf # 导入TF 库
from tensorflow import keras # 导入TF 子库
from tensorflow.keras import layers, optimizers, datasets # 导入TF 子库

(x, y), (x_val, y_val) = datasets.mnist.load_data() # 加载数据集
x = 2*tf.convert_to_tensor(x, dtype=tf.float32)/255.-1 # 转换为张量，缩放到-1~1
y = tf.convert_to_tensor(y, dtype=tf.int32) # 转换为张量
y = tf.one_hot(y, depth=10) # one-hot 编码
print(x.shape, y.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((x, y)) # 构建数据集对象
train_dataset = train_dataset.batch(512) # 批量训练
```
- 网络搭建
```
model = keras.Sequential([ # 3 个非线性层的嵌套模型
	layers.Dense(256, activation='relu'),
	layers.Dense(128, activation='relu'),
	layers.Dense(10)])
```
- 模型训练
```
with tf.GradientTape() as tape: # 构建梯度记录环境
	# 打平，[b, 28, 28] => [b, 784]
	x = tf.reshape(x, (-1, 28*28))
	# Step1. 得到模型输出output
	# [b, 784] => [b, 10]
	out = model(x)
	# step2, 计算mse损失
	loss = mse(out,y)
	# Step3. 计算参数的梯度 w1, w2, w3, b1, b2, b3
	grads = tape.gradient(loss, model.trainable_variables)
```

