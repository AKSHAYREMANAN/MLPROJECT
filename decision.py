#     Decision Tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
import pandas as pd

dataset = pd.read_csv(r"Laptop_Data.csv")
print(dataset)

x = dataset.iloc[:,[1,2]].values
y = dataset.iloc[:,3].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from  sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
df2 = pd.DataFrame(y_test,y_pred)
print(df2)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy*100,'%')