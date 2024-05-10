#     k-Means Algorithm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datasets = pd.read_csv(r"manufacturing.csv")
df = pd.DataFrame(datasets)
print(df)

x = datasets.iloc[:,[3,4]].values

# from sklearn optimal number of cluster using the elbow method
from sklearn.cluster import KMeans
Wcss_list = []
# Initializing the list for the values of WCSS
# #Using for loop for iterations fro
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    Wcss_list.append(kmeans.inertia_)
print(Wcss_list)
plt.plot(range(1,11),Wcss_list)
plt.title('the elobw method graph')
plt.xlabel('number of cluster(k)')
plt.ylabel('wcss_list')
plt.show()

kmeans =KMeans(n_clusters=5,init='k-means++',random_state=42)
y_pred = kmeans.fit_predict(x)
print(y_pred)

