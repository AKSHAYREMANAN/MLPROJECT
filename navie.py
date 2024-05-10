import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
data_set = pd.read_csv(r"diabetes.csv")
df = pd.DataFrame(data_set)
print(df)

x = data_set.iloc[:,0:8].values
y = data_set.iloc[:,8].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
df2 = pd.DataFrame(y_test,y_pred)
print(df2)

from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

accuracy = accuracy_score(y_test,y_pred)
print('Accuracy: %.2f'%(accuracy*100))