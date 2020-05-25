from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

boston = load_boston()
data = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
data['target'] = boston['target']
print(data)

##data set preprocessing. Check if is there any value in the data set
print(data.isnull().sum())

sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.distplot(data['target'], bins=30)

## Prepare the coorelation matrix
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM']
target = data['target']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    x = data[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('target')

a = np.ones(4)
print(a)
b = np.zeros((4, 2))
print(b)
c = np.c_[a, b]
print(c)

# split the train test
X = pd.DataFrame(np.c_[data['LSTAT'], data['RM']], columns=['LSTAT', 'RM'])
Y = data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#support_vector = SVR(kernel='rbf')
#support_vector.fit(X_train, Y_train)
#y_train_predict = support_vector.predict(X_train);

linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, Y_train)
y_train_predict = linear_reg_model.predict(X_train);

rmsq = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
rsq = r2_score(Y_train, y_train_predict)

print('Training data - root mean square error =>', rmsq)
print('Training data - r square =>', rsq)

#y_test_predict = support_vector.predict(X_test)
y_test_predict = linear_reg_model.predict(X_test)

rmsq = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
rsq = r2_score(Y_test, y_test_predict)

print('Testing data - root mean square error =>', rmsq)
print('Testing data - r square =>', rsq)

# Serialize and save the model
#from sklearn.externals import joblib
import joblib
joblib.dump(linear_reg_model, 'housing_model.pkl')

# Serialize the columns
housing_model_columns = list(X_train.columns)
joblib.dump(housing_model_columns,'housing_model_columns.pkl')


#query_data = pd.DataFrame([{"LSTAT": 13.27, "RM": 6.009}])
#prediction = support_vector.predict(query_data)

query = {'LSTAT': [13.27,20.45], 'RM': [6.009,6.377]}
query_data = pd.DataFrame(query, columns=['LSTAT','RM'])

prediction = joblib.load('housing_model.pkl').predict(query_data)
print('predicted price based oon pkl',prediction)

#plt.show()

