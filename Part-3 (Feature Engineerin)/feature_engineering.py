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
