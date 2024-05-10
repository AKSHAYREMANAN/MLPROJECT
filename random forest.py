#    Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datasets = pd.read_csv(r"Laptop_Data.csv")
df = pd.DataFrame(datasets)
print(df)

x = datasets.iloc[:,[1,2]].values
y = datasets.iloc[:,3].values

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#feature Scaling

from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

#Fitting Decision Tree classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 10,criterion='entropy')
classifier.fit(x_train,y_train)

#Predicting the test set result
y_pred = classifier.predict(x_test)
print("------------PREDICTION----------")
df2 = pd.DataFrame(y_test,y_pred)
print(df2.to_string())

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy*100,'%')

#Predicting the profit for the data;
x_test1 = [[19,19000]]
x_test1 = st_x.transform(x_test1)
y_pred2 = classifier.predict(x_test1)

print(y_pred2)