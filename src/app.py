import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

def scaler(X_train_, X_test_, X):
  X_train = X_train_.copy()
  X_test = X_test_.copy()
  scaler = StandardScaler()
  # Train
  X_train_scaled = pd.DataFrame(
             scaler.fit_transform(X_train[X]),
             columns=scaler.get_feature_names_out(),
             index = X_train.index)
  X_train_scaled = X_train_scaled.join(X_train[list(set(X_train.columns)  - set(X))])
  # Test
  X_test_scaled = pd.DataFrame(
      scaler.transform(X_test[X]),
      columns = scaler.get_feature_names_out(),
      index = X_test.index)
  X_test_scaled = X_test_scaled.join(X_test[list(set(X_test.columns)  - set(X))])
  X_test_scaled = X_test_scaled[X_train_scaled.columns]
  return X_train_scaled, X_test_scaled

def Elastic_gridcv(X_train, y_train):
    model =  ElasticNet(random_state=42)
    hyperparams = {"alpha" :  [0.0001, 0.01, 1, 10],
                   "l1_ratio" :  np.linspace(0,1,35),
                   "max_iter": [5, 10, 50, 100, 150],
                   "selection": ['cyclic', 'random'],
                   "tol": [1e-3, 1e-5, 1e-7, 1e-10],}
    cv = KFold(n_splits=5, shuffle=True, random_state=42) # replicables...
    grid_search = GridSearchCV(estimator=model,
                               param_grid=hyperparams,
                               cv=cv,
                               scoring= 'neg_mean_absolute_error',)
    grid_result = grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

url = 'https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv'
df = pd.read_csv(url)
df.head(5)

df.isnull().value_counts()

df.drop(columns = ['COUNTY_NAME', 'STATE_NAME'], inplace=True)

df.head(5)

for var in df.columns:
  print(var)

df[['anycondition_prevalence','COPD_prevalence','% Two or more races','% Hawaiian/PI-alone','% Asian-alone','% NA/AI-alone','% Black-alone','% White-alone','80+ y/o % of total pop','diabetes_prevalence', 'Heart disease_prevalence']].corr()

X = df[['anycondition_prevalence','COPD_prevalence','% Asian-alone','% Black-alone','% White-alone','Heart disease_prevalence']]
y = df['diabetes_prevalence']

nums = ['anycondition_prevalence','COPD_prevalence','% Asian-alone','% Black-alone','% White-alone','Heart disease_prevalence']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=123)
X_train_ml, X_test_ml = scaler(X_train, X_test, nums)

mde=Elastic_gridcv(X_train_ml, y_train)
preds = mde.predict(X_test_ml)
mean_squared_error(y_test, preds)

mde

print(mean_absolute_error(y_test, preds))
