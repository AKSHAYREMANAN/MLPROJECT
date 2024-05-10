#  Principal Component Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv(r"WineQT.csv")
print(dataset)
X = dataset.iloc[:, 0:12].values
y = dataset.iloc[:, 12].values
dfx=pd.DataFrame(X)
dfy=pd.DataFrame(y)
print("X-Data")
print(dfx.to_string())
print("Y-Data")
print(dfy.to_string())

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
x_train = pca.fit_transform(x_train)
x_test  = pca.transform(x_test)

explained_variance = pca.explained_variance_ratio_

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0,)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)