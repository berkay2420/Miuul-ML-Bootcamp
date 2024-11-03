#################################
####  UNSUPERVISED LEARNING  ####
#################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)
df = pd.read_csv("USArrests.csv", index_col=0)
df.head(3)

df.info()
df.shape
df.dtypes
df.isnull().sum()
df.describe().T

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)
df[0:5]

kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_

kmeans.inertia_
#Sum of squared distances of samples to their closest cluster center,
# weighted by the sample weights if provided.

### finding optimal value for k ###

kmeans = KMeans()
ssd = [] #sum squared distances
K = range(1,30)

for k in K:
  kmeans = KMeans(n_clusters=k).fit(df)
  ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı k degerlerine karsilik SSD/SSE/SSR")
plt.ylabel("Optimum küme sayisi için elbow yöntemi")
plt.show()


kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_ #optimum k

#### Final Model ####

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters = kmeans.labels_

df = pd.read_csv("USArrests.csv", index_col=0)

df["Cluster"] = clusters
df.head(3)


df["Cluster"].value_counts()

df[df["Cluster"] ==1]
df[df["Cluster"] ==5]

df.groupby("Cluster").agg(["count","mean","median"])

df.to_csv("clusters.csv")