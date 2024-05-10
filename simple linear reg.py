#         Simple Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_set = pd.read_csv(r"C:\ml data\simplelinearregression.csv")
print(data_set)

df = pd.DataFrame(data_set)
print(df)

x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 1].values
df2 = pd.DataFrame(x)
df3 = pd.DataFrame(y)
print(df2)
print(df3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
x_pred = regressor.predict(x_train)
print("Prediction result on Test Data")
y_pred = regressor.predict(x_test)
dfs = pd.DataFrame(x_test)
print(dfs)

df3 = pd.DataFrame(y_test, y_pred)
print(df3.to_string())

plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,x_pred,color="red")
plt.title("Age vs premium (Training Dataset)")
plt.xlabel("Age")
plt.ylabel("Premium")
plt.show()

plt.scatter(x_test, y_test, color="blue")
plt.plot(x_train, x_pred, color="red")
plt.title("Age vs premium (Test Dataset)")
plt.xlabel("Age")
plt.ylabel("Premium")
plt.show()

print("Mean")
print(df['Premium'].mean())
from sklearn import metrics
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
r2_score = regressor.score(x_test,y_test)
print(r2_score*100,'%')
y_pred2 = regressor.predict([[18]])
print("exp 3.5")
print(y_pred2)
#print(type(x_test))
arr=np.array([[22],[23]])
print("arr \n")
print(arr)
y_pred3 = regressor.predict(arr)
print(y_pred3)





