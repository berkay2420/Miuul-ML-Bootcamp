from itertools import dropwhile
from nis import cat
from tabnanny import check
from tokenize import String
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axis, pyplot as plt
import missingno as msno
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sympy import Q, false

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' %x)
pd.set_option('display.width',500)

def load_application_train():
  data = pd.read_csv("application_train.csv")
  return data

df = load_application_train()
df.head()

def load():
  data = pd.read_csv("titanic.csv")
  return data

df =load()
df.head()


#####################
#### Outliers #######
#####################


#### Finding Outlier Values ####

sns.boxplot(x=df["Age"])
plt.show()

##IQR (Interquartile Range)
q1 = df["Age"].quantile(0.25) 

q3 = df["Age"].quantile(0.75) 

iqr = q3 - q1

up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

df[(df["Age"] < low) | (df["Age"] > up )] # ----> Outliers

df[(df["Age"] < low) | (df["Age"] > up )].index

### Checking for any outliers
df[(df["Age"] < low) | (df["Age"] > up )].any(axis=None)

df[(df["Age"] < low) | (df["Age"] > up )].any(axis=None)


###
def check_for_outliers_v1(dataframe, variable):
  q1 = dataframe[variable].quantile(0.25)
  q3 = dataframe[variable].quantile(0.75) 

  iqr = q3 - q1

  up = q3 + 1.5 * iqr
  low = q1 - 1.5 * iqr

  return dataframe[(dataframe[variable] < low) | (dataframe[variable] > up)].any(axis=None)

check_for_outliers_v1(df, "PassengerId")

####
def outlier_tresholds(dataframe, col_name, q1=0.25, q3=0.75):
  quartile1 = dataframe[col_name].quantile(q1)
  quartile3 = dataframe[col_name].quantile(q3)
  interquartile_range = quartile3 - quartile1
  up_limit = quartile3 + 1.5 * interquartile_range
  low_limit = quartile1 - 1.5 * interquartile_range 
  return  low_limit, up_limit

outlier_tresholds(df, "PassengerId")
outlier_tresholds(df, "Fare")
outlier_tresholds(df, "Age")

low, up = outlier_tresholds(df, "Age")

def check_outlier(dataframe, col_name):
  low_limit, up_limit = outlier_tresholds(dataframe, col_name)
  
  if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit  )].any(axis=None):
    return True
  else:
    return False

###
def check_cols(dataframe):
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int","float"] ]

  for col in num_cols:
    if check_outlier(dataframe, col):
      print(f"{col} column have outliers")
    else:
      print(f"{col} column doesn't have outliers")

check_cols(df)

df2 = load_application_train()

check_cols(df2)

####
def grab_cols(dataframe, cat_th=10, car_th=20):
  """
  Returns categorical, numerical and and high cardinality categorical variables from the given dataframe.
  Args:
      dataframe (dataframe):
        The dataframe from which to extract variable names.
      cat_th (int, float, optional): 
        Treshold value for low cardinality numerical variables.
        Defaults to 10.
      car_th (int, float, optional):
        Treshold value for low cardinality categorical variables.
        Defaults to 20.
  Returns:
      cat_cols: (list)
        Categorical variables list
      num_cols: (list)
        Nurmerical variables list
      cat_but_car: (list)
        Low cardinality categorical variables list
  Notes:
  -cat_cols = num_cols + cat_but_car = total variables
  -cat_cols includes num_but_cat 
  """

  # cat_cols, cat_but_car
  cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
  num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and 
                 dataframe[col].dtypes != "O"] 
  cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > cat_th and
                 dataframe[col].dtypes == "O"]
  cat_cols = cat_cols + num_but_cat
  cat_cols = [col for col in cat_cols if col not in cat_but_car]

  #num_cols
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
  num_cols = [col for col in num_cols if col not in num_but_cat ]

  print(f"Observations:{dataframe.shape[0]}")
  print(f"Variables:{dataframe.shape[1]}")
  print(f"categorical_cols:{len(cat_cols)}")
  print(f"numerical_cols:{len(num_cols)}")
  print(f"categorical_but_cardinite:{len(cat_but_car)}")
  print(f"numerical_but_cardinite:{len(num_but_cat)}")

  return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_cols(df2)

cat_cols, num_cols, cat_but_car = grab_cols(df)

####
def grab_outliers(dataframe, col_name, index=False):
  low, up = outlier_tresholds(dataframe, col_name)
  outlier_count = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up  )].shape[0]

  if outlier_count > 10:
    print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].head(5))
  else:
    print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)])
  
  if index:
    outlier_index = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].index
    return outlier_index
  
grab_outliers(df, "Fare", index=True)

age_index = grab_outliers(df, "Age", index=True)

df.loc[age_index, "Age"]

df.iloc[age_index]  # outlier rows from age index

#### Deleting Outliers ####
low, up = outlier_tresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low)|(df["Fare"] > up))].shape # normal ones NOT OUTLIERS

###
def remove_outlier(dataframe, col_name):
  low_limit, up_limit = outlier_tresholds(dataframe, col_name)
  df_without_outliers = dataframe[~((dataframe[col_name] < low_limit)|(dataframe[col_name] > up_limit))]
  return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_cols(df)

num_cols = [col for col in num_cols if col not in "PassangerId"]

df.shape

for col in num_cols:
  new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]

#### Re-assignment with tresholds ####

low, up = outlier_tresholds(df, "Fare")

df[((df["Fare"] < low)|(df["Fare"] > up))]["Fare"]

####
df.loc[((df["Fare"] < low)|(df["Fare"] > up)), "Fare" ]

df.loc[(df["Fare" > up], "Fare")] = up # up limitinden büyük olanları up değeri ile değiştirdik. 
                                       # yani örneğin 100den büyük olan tüm değerleri 100 yaptık
  
def replace_with_tresholds(dataframe, variable):
  low_limit, up_limit = outlier_tresholds(dataframe, variable)
  dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
  dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

df = load()

cat_cols, num_cols, cat_but_car = grab_cols(df)

num_cols = [col for col in num_cols if col not in "PassangerId"]

df.shape

for col in num_cols:
  print(col, check_outlier(df, col))


for col in num_cols:
  replace_with_tresholds(df, col)

for col in num_cols:
  print(col, check_outlier(df, col))
  ## outliers have been deleted

#### Recap ####
df = load()

outlier_tresholds(df, "Age" ) #Finding outlier threshold for variable
check_outlier(df, "Age") # checking if there is ant outlier in variable
grab_outliers(df, "Age", index=True) # displaying outliers 

df_without_outliers = remove_outlier(df, "Age").shape #outliers removed version of database

replace_with_tresholds(df, "Age") #replacing outliers with threshold values

##################################
##### Local Outlier Factor ######
##################################
df = sns.load_dataset("diamonds")
df = df.select_dtypes(["int64","float64"])
df = df.dropna()
df.head()

for col in df.columns:
  print(col, check_outlier(df, col ))


low, up = outlier_tresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

###
clf = LocalOutlierFactor(n_neighbors=20) #Unsupervised Outlier Detection using the Local Outlier Factor (LOF).
# By comparing the local density of a sample to the local densities of its neighbors, 
# one can identify samples that have a substantially lower density than their neighbors. 
# These are considered outliers.
# Works on every column every data



clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_ #Lower values indicate that the point is considered more of an outlier.
df_scores[0:5]
# df_scores = -df_scores

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,20], style='.-')
plt.show()

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,50], style='.-')
plt.show()

th = np.sort(df_scores)[3] #treshold

df[df_scores < th] #outliers

df[df_scores < th].index

df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)
