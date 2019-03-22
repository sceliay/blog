---
title: pandas
date: 2019-03-22 17:12:44
tags: [pandas, python]
---

一些pandas的语法

1. 读取与存储
`read_csv`, `to_csv`, `read_pickle`,`to_pickle`

2. 缺损值填充
- `data = data.fillna('-1', )`
- `data = data.fillna(data.mean(), )`
- 填充[]:
```
nan_index = data['title_features'][data['title_features'].isnull()].index
data['title_features'].iloc[nan_index] = [[]]
```

3. 计算缺损值个数：
`Nan_num = data.shape[1]-data(axis=1)`

4. 去掉缺损值过多的行：
`data.drop(Nan_num[Nan_num>256.index.tolist(),inplace=True)`