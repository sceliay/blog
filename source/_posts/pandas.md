---
categories: Python
title: Pandas学习笔记
date: 2019-03-22 17:12:44
tags: [Python, Pandas]
---

记录一些pandas的语法
[官方文档](http://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_sql.html), [pandas vs. SQL](http://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_sql.html), [教程](https://www.cnblogs.com/en-heng/p/5630849.html)


1. 读取与存储
`read_csv`, `to_csv`, `read_pickle`,`to_pickle`

2. 缺损值填充
- `data = data.fillna('-1', )`
- `data = data.fillna(data.mean(), )`
- 填充[]: 不过，尽量不要用DataFrame存储list
```
nan_index = data[data.isnull()].index
data.loc[nan_index,] = [[]]
```

3. 计算缺损值个数：
`Nan_num = data.shape[1]-data.count(axis=1)`

4. 去掉缺损值过多的行：
`data.drop(Nan_num[Nan_num>256.index.tolist(),inplace=True)`
drop函数默认删除行，列需要加`axis = 1`

5. df.info()查看数据type

6. [数据转换](https://juejin.im/post/5acc36e66fb9a028d043c2a5)
- astype()强制转换，仅返回数据的副本而不原地修改。

- 自定义转换：
```
def convert_currency(val):
    """
    Convert the string number value to a float
     - Remove $
     - Remove commas
     - Convert to float type
    """
    new_val = val.replace(',','').replace('$', '')
    return float(new_val)

df['2016'].apply(convert_currency)
```

- 使用lambda进行转换
`df['2016'].apply(lambda x: x.replace('$', '').replace(',', '')).astype('float')`

- 使用to_numeric进行转换
`pd.to_numeric(df['Jan Units'], errors='coerce').fillna(0)`

7. [选择数据](https://morvanzhou.github.io/tutorials/data-manipulation/np-pd/3-2-pd-indexing/)
- `loc`根据标签
- `iloc`根据序列
- `ix`混合

8. [数据合并](https://blog.csdn.net/stevenkwong/article/details/52528616)

9. [统计个数](https://blog.csdn.net/waple_0820/article/details/80514073)
- `df.groupby(['id'],as_index=False)['id'].agg({'cnt':'count'})`
- `df['id'].value_counts()`

10. datetime
- `pd.to_datetime(df)`：将str转换为datetime
- `df.dt.year`: 获得datetime数据中的year
- `df.map(lambda x:x.strftime('%Y'))`

11. [对行和列的操作](https://blog.csdn.net/xiaodongxiexie/article/details/53108959)

12. [去掉含nan的行/列](https://amberwest.github.io/2019/03/05/%E5%88%A0%E9%99%A4DataFrame%E4%B8%AD%E6%9F%90%E5%88%97%E5%80%BC%E4%B8%BANaN%E7%9A%84%E8%AE%B0%E5%BD%95-%E8%A1%8C/)

13. 删去重复列
`DataFrame.drop_duplicates(subset=None, keep='first', inplace=False)`
- `subset`: 指定特定的列，默认全部
- `keep: {‘first’, ‘last’, False}, default ‘first’`: 删除重复项，并保留，默认第一次出现的
