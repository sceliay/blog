---
categories: Machine Learning
title: AI for Medicine(2)
date: 2020-04-28 19:32:47
tags: Machine Learning
---

<script type="text/javascript"
   src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# AI for Medical Prognosis
1. Linear model
```
# Import the module 'LinearRegression' from sklearn
from sklearn.linear_model import LinearRegression

# Create an object of type LinearRegression
model = LinearRegression()

# Import the load_data function from the utils module
from utils import load_data

# Generate features and labels using the imported function
X, y = load_data(100)

model.fit(X, y)
```

	- Logistic Regression: `sklearn.linear_model.LogisticRegression`

2. Risk score
	- Atrial fibrillation: Chads-vasc score
	- Liver disease: MELD score
	- Heart disease: ASCVD score

3. Measurements
	- $$ c-index = (conordant-0.5\*ties)/permission $$

4. Decision Tree
	```
	import shap
	import sklearn
	import itertools
	import pydotplus
	import numpy as np
	import pandas as pd
	import seaborn as sns
	import matplotlib.pyplot as plt
	from IPython.display import Image 
	from sklearn.tree import export_graphviz
	from sklearn.externals.six import StringIO
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.experimental import enable_iterative_imputer
	from sklearn.impute import IterativeImputer, SimpleImputer
	#
	# We'll also import some helper functions that will be useful later on.
	from util import load_data, cindex
		from sklearn.tree import DecisionTreeClassifier
		dt = DecisionTreeClassifier()
		dt.fit(X,y)
		dt = DecisionTreeClassifier(criterion='entropy',
	                            max_depth=10,
	                            min_samples_split=2
	                           )
	```
	- 控制树的深度以处理overfitting
	```
	dt_hyperparams = {
	    'max_depth':3,
	    'min_samples_split':2
	}
	dt_reg = DecisionTreeClassifier(**dt_hyperparams, random_state=10)
	dt_reg.fit(X_train_dropped, y_train_dropped)
	y_train_preds = dt_reg.predict_proba(X_train_dropped)[:, 1]
	y_val_preds = dt_reg.predict_proba(X_val_dropped)[:, 1]
	print(f"Train C-Index: {cindex(y_train_dropped.values, y_train_preds)}")
	print(f"Val C-Index (expected > 0.6): {cindex(y_val_dropped.values, y_val_preds)}")
	#
	# print tree
	from sklearn.externals.six import StringIO
	dot_data = StringIO()
	export_graphviz(dt_reg, feature_names=X_train_dropped.columns, out_file=dot_data,  
	                filled=True, rounded=True, proportion=True, special_characters=True,
	                impurity=False, class_names=['neg', 'pos'], precision=2)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
	Image(graph.create_png()) 
	```
	- Random forests
	```
	rf = RandomForestClassifier(n_estimators=100, random_state=10)
	rf.fit(X_train_dropped, y_train_dropped)
	#
	def random_forest_grid_search(X_train_dropped, y_train_dropped, X_val_dropped, y_val_dropped):
		#
	    # Define ranges for the chosen random forest hyperparameters 
	    hyperparams = {
	        'n_estimators': [1,2,3],
	        'max_depth': [4,5,6],
	        'min_samples_leaf': [1,2,3],
	    }    
	    fixed_hyperparams = {
	        'random_state': 10,
	    }    
	    rf = RandomForestClassifier
	    best_rf, best_hyperparams = holdout_grid_search(rf, X_train_dropped, y_train_dropped,
	                                                    X_val_dropped, y_val_dropped, hyperparams,
	                                                    fixed_hyperparams)
	    print(f"Best hyperparameters:\n{best_hyperparams}")
	    #
	    y_train_best = best_rf.predict_proba(X_train_dropped)[:, 1]
	    print(f"Train C-Index: {cindex(y_train_dropped, y_train_best)}")
	    #
	    y_val_best = best_rf.predict_proba(X_val_dropped)[:, 1]
	    print(f"Val C-Index: {cindex(y_val_dropped, y_val_best)}")
	    #
	    # add fixed hyperparamters to best combination of variable hyperparameters
	    best_hyperparams.update(fixed_hyperparams)
	    #
	    return best_rf, best_hyperparams
	#
	best_rf, best_hyperparams = random_forest_grid_search(X_train_dropped, y_train_dropped, X_val_dropped, y_val_dropped)
	```

5. Ensembling 
	- desision tree 
	- gradient boosting
	- xgboost
	- lightGBM

6. NAN
	```
	df_booleans = df.isnull()
	df_booleans.any()
	```

6. 	Imputation
	- mean
	```
	from sklearn.impute import SimpleImputer
	mean_imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
	mean_imputer.fit(df)
	```
	- regression：age 和 BP 具有线性关系
	```
	from sklearn.experimental import enable_iterative_imputer
	from sklearn.impute import IterativeImputer
	reg_imputer = IterativeImputer()
	reg_imputer.fit(df)
	nparray_imputed_reg = reg_imputer.transform(df)
	```

7. SHAP
```
explainer = shap.TreeExplainer(rf_imputed)
i = 0
shap_value = explainer.shap_values(X_test.loc[X_test_risk.index[i], :])[1]
shap.force_plot(explainer.expected_value[1], shap_value, feature_names=X_test.columns, matplotlib=True)
```