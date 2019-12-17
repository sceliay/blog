---
categories: Python
title: Python
date: 2018-12-20 20:07:11
tags: Python
---

python中的一些语法&函数。

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

9. 删除list中的元素
- remove: 删除单个元素，删除首个符合条件的元素，按值删除
举例说明:
```
str=[1,2,3,4,5,2,6]
str.remove(2)
str
# [1, 3, 4, 5, 2, 6]
```

- pop:  删除单个或多个元素，按位删除(根据索引删除)
```
str=[0,1,2,3,4,5,6]
str.pop(1)   #pop删除时会返回被删除的元素
# 1
str
# [0, 2, 3, 4, 5, 6]
str2=['abc','bcd','dce']
str2.pop(2)
# 'dce'
str2
# ['abc', 'bcd']
```

- del：它是根据索引(元素所在位置)来删除
```
str=[1,2,3,4,5,2,6]
del str[1]
str
# [1, 3, 4, 5, 2, 6]
str2=['abc','bcd','dce']
del str2[1]
str2
# ['abc', 'dce']
```

除此之外，del还可以删除指定范围内的值。
```
str=[0,1,2,3,4,5,6]
del str[2:4]  #删除从第2个元素开始，到第4个为止的元素(但是不包括尾部元素)
str
# [0, 1, 4, 5, 6]
```
del 也可以删除整个数据对象(列表、集合等)
```
str=[0,1,2,3,4,5,6]
del str
str         #删除后，找不到对象
```

10. [LabelEncoder](https://blog.csdn.net/quintind/article/details/79850455)
LabelEncoder是用来对分类型特征值进行编码，即对不连续的数值或文本进行编码。其中包含以下常用方法：
`fit(y)` ：fit可看做一本空字典，y可看作要塞到字典中的词。 
`fit_transform(y)`：相当于先进行fit再进行transform，即把y塞到字典中去以后再进行transform得到索引值。 
`inverse_transform(y)`：根据索引值y获得原始数据。 
`transform(y)` ：将y转变成索引值。
```
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"]) 
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
```

11. [数据预处理](https://www.cnblogs.com/chaosimple/p/4153167.html)

12. [排列组合](https://www.cnblogs.com/aiguiling/p/8594023.html)
- `product` 笛卡尔积　　（有放回抽样排列）
- `permutations` 排列　　（不放回抽样排列）
- `combinations` 组合,没有重复　　（不放回抽样组合）
- `combinations_with_replacement` 组合,有重复　　（有放回抽样组合）
- eg: 
```
import itertools
itertools.product('ABCD', repeat = 2)
```
- combinations和permutations返回的是对象地址，iterators（迭代器），可以用list()转换为list

13. 删除内存
del 可以删除多个变量，del a,b,c,d
办法：
import gc （garbage collector）
del a
gc.collect()
马上内存就释放了。

14. `enumerate()` 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
```
>>>seasons = ['Spring', 'Summer', 'Fall', 'Winter']
>>> list(enumerate(seasons))
[(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
>>> list(enumerate(seasons, start=1))       # 下标从 1 开始
[(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

14. 将lst = [ [1, 2, 3], [2, 1, 3], [8, 4, 3] ]，变为[1, 2, 3, 2, 1, 3, 8, 4, 3]
- `myList = [x for j in lst for x in j]`
- ```
   from itertools import chain
   mylist = list(chain(*lst))
  ```

15. [namedtuple](https://www.cnblogs.com/herbert/p/3468294.html)
namedtuple创建一个和tuple类似的对象，而且对象拥有可以访问的属性。这对象更像带有数据属性的类，不过数据属性是只读的。
eg: `TPoint = namedtuple('TPoint', ['x', 'y'])` 创建一个TPoint类型，而且带有属性x, y.

16. [isinstance](http://www.runoob.com/python/python-func-isinstance.html)
用来判断一个对象是否是一个已知的类型，类似 type()。
`isinstance()` 与 `type()` 区别：
- `type()` 不会认为子类是一种父类类型，不考虑继承关系。
- `isinstance()` 会认为子类是一种父类类型，考虑继承关系。
使用：`isinstance(object, classinfo)`