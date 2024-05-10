#      Polynomial Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_set = pd.read_csv(r"xy data.csv")
print(data_set)

#Extracting Independent and dependent Variable

x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 1].values
df1 = pd.DataFrame(x)
df2 = pd.DataFrame(y)

from sklearn.linear_model import LinearRegression
lin_regs = LinearRegression()
lin_regs.fit(x,y)

#Building the Polynomial regression model:

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x,y,color="blue")

#Visulaizing the result for Polynomial
#
plt.scatter(x,y,color="blue")
plt.plot(x,lin_regs.predict(x), color="red")
plt.title("Bluff detection model(Linear Regression)")

plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.scatter(x,y,color="blue")
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color="red")
plt.title("Bluff detection model(Polynomial Regression)")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Predicting the final result with the Linear Regression model:

lin_pred = lin_regs.predict([[47]])
print(lin_pred)

#Predicting the final result with the Polynomial Regression model:

poly_pred = lin_reg_2.predict(poly_reg.fit_transform([[36]]))
print(poly_pred)