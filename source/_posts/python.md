---
categories: Notes
title: python
date: 2018-12-20 20:07:11
tags: Python
---

渣渣程序员记录一下python中的一些语法&函数。

1. 原数+1: `x += 1`

2. 字符串与数字的转换：
- `str(2) = '2'`
- `int('2') = 2`

3. 字符串的提取：
- `str.strip([chars])`
   - 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
- `str.split(str="", num=string.count(str))`
   - 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则分隔 num+1 个子字符串

4. 字符串的拼接：
- 用%拼接：
    - `str = 'There are %s, %s, %s on the table.' % (fruit1,fruit2,fruit3)`
    - `str = 'There are %(fruit1)s,%(fruit2)s,%(fruit3)s on the table' % {'fruit1':fruit1,'fruit2':fruit2,'fruit3':fruit3} `
    - `%s` str, `%d` int, `%f` float, `%.2f` 两位小数float, `%8s` 8位占位符，`%-8s` 左对齐 
- 用join()拼接：
    ```
    temp = ['There are ',fruit1,',',fruit2,',',fruit3,' on the table']
    ''.join(temp)
    ```
- 用format()拼接：
    ```
    str = 'There are {2}, {1}, {0} on the table'
    str.format(fruit1,fruit2,fruit3) #fruit1出现在0的位置
    ```
    ```
    str = 'There are {fruit1}, {fruit2}, {fruit3} on the table'
	  str.format(fruit1=fruit1,fruit2=fruit2,fruit3=fruit3)
	 ```

5. 遍历list：
```
from collections import Counter
Counter([1,2,2,2,2,3,3,3,4,4,4,4])

# Counter({2: 4, 4: 4, 3: 3, 1: 1})
```

6. numpy.random.uniform(low,high,size): [参考](https://blog.csdn.net/u013920434/article/details/52507173)
low: 采样下界，float类型，默认值为0；
high: 采样上界，float类型，默认值为1；
size: 输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出m*n*k个样本，缺省时输出1个值。
返回值：ndarray类型，其形状和参数size中描述一致。

7. OrderedDict(): [参考](https://www.cnblogs.com/gide/p/6370082.html)
很多人认为python中的字典是无序的，因为它是按照hash来存储的，但是python中有个模块collections(英文，收集、集合)，里面自带了一个子类OrderedDict，实现了对字典对象中元素的排序。

8. zip():[参考](http://www.runoob.com/python/python-func-zip.html)
zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
```
a = [1,2,3]
b = [4,5,6]
c = [4,5,6,7,8]
zipped = zip(a,b)     # 打包为元组的列表
# [(1, 4), (2, 5), (3, 6)]
zip(a,c)              # 元素个数与最短的列表一致
# [(1, 4), (2, 5), (3, 6)]
zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
# [(1, 2, 3), (4, 5, 6)]
```