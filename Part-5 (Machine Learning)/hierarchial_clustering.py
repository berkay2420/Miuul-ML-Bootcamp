from lib2to3.fixes import fix_dict
from cv2 import KMEANS_PP_CENTERS
from matplotlib.lines import lineStyles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import lasso_path
from sklearn.preprocessing import MinMaxScaler
from sympy import Min
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)
df = pd.read_csv("USArrests.csv", index_col=0)
df.head(3)

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")

plt.figure(figsize=(10,5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimi")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size=10)
plt.show()

## better graph
plt.figure(figsize=(10,5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimi")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

## finding cluster value

plt.figure(figsize=(7,5))
plt.title("Dendograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.6, color='r', linestyle='--')
plt.axhline(y=0.5, color='b', linestyle='--')
plt.show()


### Final Model

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)

df = pd.read_csv("USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

kmeans = KMeans()
ssd = [] #sum squared distances
K = range(1,30)

for k in K:
  kmeans = KMeans(n_clusters=k).fit(df)
  ssd.append(kmeans.inertia_)

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(df)
elbow.show()

df = pd.read_csv("USArrests.csv", index_col=0)
df = sc.fit_transform(df)
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans_clusters = kmeans.labels_

df = pd.read_csv("USArrests.csv", index_col=0)
df["hi_cluster_no"] = clusters
df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = kmeans_clusters
df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1

df.head(3)

df["hi_cluster_no"].value_counts()

df["kmeans_cluster_no"].value_counts()

df[df["hi_cluster_no"] == df["kmeans_cluster_no"]]