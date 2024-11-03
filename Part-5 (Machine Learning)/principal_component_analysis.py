from cProfile import label
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
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
df = pd.read_csv("hitters.csv")
df.head(3)

num_cols = [col for col in df.columns if df[col].dtype != "O" and "Salary" not in col] 

df = df[num_cols]
df.dropna(inplace=True)
df.shape

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)


#### Finding Optimum Component Value ####
pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Varience Ratio")
plt.show()

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

############################################
####   PRINCIPAL COMPONENT REGRESSION   ####
############################################

df = pd.read_csv("hitters.csv")
df.shape

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]

pd.DataFrame(pca_fit, columns=["PCA1","PCA2","PCA3"]).head()

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),df[others]], axis=1)
final_df.head(3)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder

def label_encoder(dataframe, binary_col):
  label_encoder = LabelEncoder()
  dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
  return dataframe

for col in ["NewLeague","Division","League"]:
  label_encoder(final_df, col)

final_df.dropna(inplace=True)

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()

rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))

cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse

cart_params = {"max_depth":range(1,11),
               "min_samples_split":range(2,20)}

from sklearn.model_selection import GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X,y)

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X,y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse

###########################
#### PCA Vısualization ####
###########################

pd.set_option('display.max_columns',None)
pd.set_option('display.width', 500)

df = pd.read_csv("breast_cancer.csv")
df.head(3)

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)

def create_pca_df(X,y):
  X = StandardScaler().fit_transform(X)
  pca = PCA(n_components=2)
  pca_fit = pca.fit_transform(X)
  pca_df = pd.DataFrame(data=pca_fit, columns=["PC1","PC2"])
  final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
  return final_df

pca_df = create_pca_df(X,y)

def plot_pca(dataframe, target):
  fig = plt.figure(figsize=(7,5))
  ax = fig.add_subplot(1,1,1)
  ax.set_xlabel("PC1", fontsize=15)
  ax.set_ylabel("PC2", fontsize=15)
  ax.set_title(f"{target.capitalize()}", fontsize=20)

  targets = list(dataframe[target].unique())
  len(targets)
  colors = ['r','b','g','y']

  for t in targets:
    indices = dataframe[target] == t
    color = random.choice(colors)
    ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    colors.remove(color)

  # for t, color in zip(targets, colors):
  #   indices = dataframe[target] == t
  #   ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
  
  ax.legend(targets)
  ax.grid()
  plt.show()

plot_pca(pca_df, "diagnosis")


################
import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)
# X must be numerical

pca_df = create_pca_df(X,y)

plot_pca(pca_df, "species")