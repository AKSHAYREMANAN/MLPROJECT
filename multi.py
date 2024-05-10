#     Multiple Linear Regression

import pandas as pd
import numpy as np
data_set=pd.read_csv(r"diabetes.csv")
print(data_set.describe())

x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 4].values
df2 = pd.DataFrame(x)
print('x=')
print(df2.to_string())
df3 = pd.DataFrame(y)
print('y=')
print(df3.to_string())

#Catgorical data

from sklearn . preprocessing import LabelEncoder, OneHotEncoder
from sklearn . compose import ColumnTransformer
labelencoder_x = LabelEncoder()

x[:, 3] = labelencoder_x . fit_transform(x[:, 3])
dt = pd.DataFrame(x)
print(dt.to_string())

# State column

ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
x = ct.fit_transform(x)
dfx = pd.DataFrame(x)

#avoiding the dummy variable trap:

x = x[:, 1:]
df4 = pd .DataFrame(x)
print(df4.to_string())

# Splitting the dataset into training and test set.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set result;
y_pred = regressor.predict(x_test)

#To compare the actual output values for X_test with the predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.to_string())

print('mean')
print(data_set.describe())

#Evaluating the Algorithm

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('root mean squred error:')
np.sqrt(metrics.mean_squared_error(y_test , y_pred))

from sklearn.metrics import r2_score

# predicting the accuracy score
score = r2_score(y_test , y_pred)
print('r2 score is', score*100, '%')
k_test =np.array ([50,32])

# # Predicting the profit for the data;
# y_p = regressor.predict(k_test)
# print(y_p)
