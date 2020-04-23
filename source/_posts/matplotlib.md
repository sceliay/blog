---
categories: Python
title: Matplotlib
date: 2020-04-23 16:16:30
tags: Python
---

主要参考：[Matplotlib Python3 数据可视化神器](https://morvanzhou.github.io/tutorials/data-manipulation/plt/)

1. 基础用法
```
import matplotlib.pyplot as plt
import numpy as np

# 使用np.linspace定义x：范围是(-1,1);个数是50. 仿真一维数据组(x ,y)表示曲线1.
x = np.linspace(-1, 1, 50)
y = 2*x + 1

# 使用plt.figure定义一个图像窗口. 使用plt.plot画(x ,y)曲线. 使用plt.show显示图像.
plt.figure()
plt.plot(x, y)
plt.show()
```

2. 画多根线条
```
# 定义曲线2
y2 = x**2

# 绘制两条曲线
plt.figure(num=3, figsize=(8, 5),)
plt.plot(x, y2)
plt.plot(x, y, color='red', linewidth=1.0, linestyle='--')
plt.show()
```

3. 调整名字和间距
使用plt.xlim设置x坐标轴范围：(-1, 2)； 使用plt.ylim设置y坐标轴范围：(-2, 3)； 使用plt.xlabel设置x坐标轴名称：’I am x’； 使用plt.ylabel设置y坐标轴名称：’I am y’：
```
plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel('I am x')
plt.ylabel('I am y')
```
添加在`plt.show()`前。

使用np.linspace定义范围以及个数：范围是(-1,2);个数是5. 使用print打印出新定义的范围. 使用plt.xticks设置x轴刻度：范围是(-1,2);个数是5.
```
new_ticks = np.linspace(-1, 2, 5)
print(new_ticks)
plt.xticks(new_ticks)
```
使用plt.yticks设置y轴刻度以及名称：刻度为[-2, -1.8, -1, 1.22, 3]；对应刻度的名称为[‘really bad’,’bad’,’normal’,’good’, ‘really good’]. 使用plt.show显示图像.
```
plt.yticks([-2, -1.8, -1, 1.22, 3],[r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
plt.show()
```

使用plt.gca获取当前坐标轴信息. 使用.spines设置边框：右侧边框；使用.set_color设置边框颜色：默认白色； 使用.spines设置边框：上边框；使用.set_color设置边框颜色：默认白色；
```
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()
```