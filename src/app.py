import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()

df['houseval'] = data.target

X = df.drop(columns=['houseval'])
y = df['houseval']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)

# Lasso
from sklearn.model_selection import GridSearchCV
robot_Lasso = Lasso()
parameters = {'alpha': [1, 0.1, 0.01, ],
              'max_iter': [10, 20, 30, 40, 50, 60],
              "tol": [0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07],
              "selection": ["random"]}
clf_Lasso = GridSearchCV(robot_Lasso, parameters,
                   cv = 5,
                   scoring='neg_mean_squared_error')
clf_Lasso.fit(X_train,y_train)

# Ridge
from sklearn.model_selection import GridSearchCV
robot_Ridge = Ridge()
parameters = {'alpha': [0.1, 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09],
              'max_iter': [1, 5, 10, 20],
              "tol": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001,
                      1e-05, 1e-06, 1e-07, 1e-08, 1e-09],
              "solver": ["auto", "svd", "cholesky", 
                         "lsqr", "sparse_cg", "sag", "saga"]}
clf_Ridge = GridSearchCV(robot_Ridge, parameters,
                   cv = 5,
                   scoring='neg_mean_squared_error')
clf_Ridge.fit(X_train,y_train)

y_pred_Lasso = clf_Lasso.predict(X_test)

y_pred_Ridge = clf_Ridge.predict(X_test)

results_df_Lasso = pd.DataFrame(clf_Lasso.cv_results_)
# Show mean and std of the negative MSE for each alpha
print(results_df_Lasso[['param_alpha', 'mean_test_score', 'std_test_score']])

results_df_Ridge = pd.DataFrame(clf_Ridge.cv_results_)
# Show mean and std of the negative MSE for each alpha
print(results_df_Ridge[['param_alpha', 'mean_test_score', 'std_test_score']])

clf_Lasso.best_estimator_

clf_Ridge.best_estimator_

mean_squared_error(y_test, y_pred_Lasso)

mean_squared_error(y_test, y_pred_Ridge)

# Escalamos los datos
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Lasso
robot_Lasso = Lasso()
parameters = {'alpha': [0.1, 0.05, 0.01, 0.001, 0.0001],
              'max_iter': [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000],
              "tol": [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001],
              "selection": ["random", "cyclic"]}
clf_Lasso = GridSearchCV(robot_Lasso, parameters,
                   cv = 5,
                   scoring='neg_mean_squared_error')
clf_Lasso.fit(X_train,y_train)

# Ridge
robot_Ridge = Ridge()
parameters = {'alpha': [4, 3, 2, 1.5, 1, 0.1, 0.01, 0.001, 0.0001],
              'max_iter': [2, 5, 10, 20],
              "tol": [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001,
                      1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10],
              "solver": ["auto", "svd", "cholesky", 
                         "lsqr", "sparse_cg", "sag", "saga"]}
clf_Ridge = GridSearchCV(robot_Ridge, parameters,
                   cv = 5,
                   scoring='neg_mean_squared_error')
clf_Ridge.fit(X_train,y_train)

y_pred_Ridge = clf_Ridge.predict(X_test)

y_pred_Lasso = clf_Lasso.predict(X_test)

mean_squared_error(y_test, y_pred_Lasso)

mean_squared_error(y_test, y_pred_Ridge)

clf_Lasso.best_estimator_

clf_Ridge.best_estimator_
