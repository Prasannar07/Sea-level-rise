import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("samples_amh_gmsl2100.csv") 
df.head()
df.isnull().sum()
X = df.iloc[:, :5].astype('float64')
y1 = df[['GMSL_RCP26']]
y2 = df[['GMSL_RCP85']]
feature_names = X.columns



X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.3, random_state=10)
scaler = StandardScaler().fit(X_train[feature_names]) 
X_train[feature_names] = scaler.transform(X_train[feature_names])
X_test[feature_names] = scaler.transform(X_test[feature_names])
df_train = y_train.join(X_train)
df_test = y_test.join(X_test)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
rfc=RandomForestRegressor(random_state=42)
param_grid = { 
            "n_estimators"      : [50,100,200],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "max_depth": [5, 10, 20],
            "bootstrap": [True, False],
            }

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_
rfc1 = RandomForestRegressor(random_state=42, bootstrap=True, max_features='auto', min_samples_split = 2, 
n_estimators= 100, max_depth=10)
rfc1.fit(X_train, y_train)
pred=rfc1.predict(X_train)
print("MSE for Random Forest on CV data: ", mean_squared_error(y_train, pred))
# obtain feature importance
feature_importance = rfc1.feature_importances_

# sort features according to importance
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0])

# plot feature importances
plt.barh(pos, feature_importance[sorted_idx], align="center")

plt.yticks(pos, np.array(feature_names)[sorted_idx])
plt.title("Feature Importance (MDI)")
plt.xlabel("Mean decrease in impurity");


X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.3, random_state=10)
scaler = StandardScaler().fit(X_train[feature_names]) 
X_train[feature_names] = scaler.transform(X_train[feature_names])
X_test[feature_names] = scaler.transform(X_test[feature_names])
df_train = y_train.join(X_train)
df_test = y_test.join(X_test)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
rfc=RandomForestRegressor(random_state=42)
param_grid = { 
            "n_estimators"      : [50,100,200],
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : [2,4,8],
            "max_depth": [5, 10, 20],
            "bootstrap": [True, False],
            }

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_
rfc1 = RandomForestRegressor(random_state=42, bootstrap=True, max_features='auto', min_samples_split = 2, 
n_estimators= 100, max_depth=10)
rfc1.fit(X_train, y_train)
pred=rfc1.predict(X_train)
print("MSE for Random Forest on CV data: ", mean_squared_error(y_train, pred))
# obtain feature importance
feature_importance = rfc1.feature_importances_

# sort features according to importance
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0])

# plot feature importances
plt.barh(pos, feature_importance[sorted_idx], align="center")

plt.yticks(pos, np.array(feature_names)[sorted_idx])
plt.title("Feature Importance (MDI)")
plt.xlabel("Mean decrease in impurity");
