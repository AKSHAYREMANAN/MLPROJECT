#    K-Nearest Neighbor Algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
data_set = pd.read_csv(r"Laptop_Data.csv")
print(data_set)
df = pd.DataFrame(data_set)
print(df)

x = data_set.iloc[:,[1,2]].values
y = data_set.iloc[: ,3].values
df2 = pd.DataFrame(x)
print(df2)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
df3 = pd.DataFrame(x_train)
print(df3)

from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)
df4 = pd.DataFrame(x_train)
print(df4)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print(y_pred)

ddf = pd.DataFrame(y_test,y_pred)
print(ddf)

accuracy = accuracy_score(y_test,y_pred)
print (accuracy*100,'%')

test = [[45,43000]]
k_test = st_x.transform(test)
pred = classifier.predict(k_test)
print(pred)