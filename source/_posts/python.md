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

