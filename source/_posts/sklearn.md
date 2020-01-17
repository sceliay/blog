---
categories: Machine Learning
title: sklearn
date: 2020-01-16 19:39:36
tags: [Machine Learning]
---
 
# Classification metrics
1. [链接](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
```
from sklearn import metrics

metrics.roc_auc_score(y_true,y_pre)

# F1 = 2 * (precision * recall) / (precision + recall)
metrics.f1_score(y_true,y_pre)
```

# Regression metrics
```
metrics.mean_absolute_error(y_true,y_pre)
metrics.mean_squared_error(y_true,y_pre)
```