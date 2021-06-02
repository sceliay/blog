---
categories: Machine Learning
title: sklearn
date: 2020-01-16 19:39:36
tags: [Machine Learning]
---
 
1. 评估函数
- [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
```
from sklearn import metrics

metrics.roc_auc_score(y_true,y_pre)

# F1 = 2 * (precision * recall) / (precision + recall)
metrics.f1_score(y_true,y_pre)
```
- Regression metrics
```
metrics.mean_absolute_error(y_true,y_pre)
metrics.mean_squared_error(y_true,y_pre)
```

2. [标准化和归一化](https://blog.csdn.net/FrankieHello/article/details/79659111)
```
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# 标准化
ss = StandardScaler() 
std_data = ss.fit_transform(data) 
origin_data = ss.inverse_transform(std_data) # 还原

# 归一化
mm = MinMaxScaler()
mm_data = mm.fit_transform(data)
origin_data = mm.inverse_transform(mm_data)
```