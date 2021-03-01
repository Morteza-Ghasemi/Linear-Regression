import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score



boston_dataset = load_boston()

print(boston_dataset.keys())
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()
boston.shape
boston.describe()
boston['MEDV'] = boston_dataset.target
boston.isnull().sum()


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()
plt.figure()
correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)



plt.figure(figsize=(20, 5))
features = ['LSTAT', 'RM']
target = boston['MEDV']
for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

 
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


#To compare the actual output values for X_test with the predicted values:
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_test_predict})


print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_test_predict))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_test_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_test_predict)))


print(lin_model.intercept_)
print(lin_model.coef_)
coeff_df = pd.DataFrame(lin_model.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
