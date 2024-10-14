from logging.handlers import DatagramHandler
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axis, pyplot as plt
import missingno as msno
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sympy import lowergamma

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' %x)
pd.set_option('display.width',500)


df = pd.read_csv("diabetes.csv")
df.head(3)
df.info
df.shape
df.columns
df.describe()

df.isnull().values.any()
df.isnull().sum()




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


cat_cols, num_cols, cat_but_car = grab_cols(df)

df[cat_cols].nunique()

def num_summary(dataframe, col):
    x = dataframe.groupby("Outcome")[col].mean()
    print(f"Mean of '{col}' by Outcome:")
    print(x)
    print("#################")

for col in num_cols:
  num_summary(df, col)

for col in num_cols:
  print(df[col].head(2))

### finding outliers
def check_for_outliers(dataframe, variable):
  q1 = dataframe[variable].quantile(0.25)
  q3 = dataframe[variable].quantile(0.75)
  iqr = q3 - q1
  low_limit = q1 - 1.5 * iqr 
  up_limit = q3 + 1.5 * iqr

  return dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None)

check_for_outliers(df, "BMI")

for col in num_cols:
  if check_for_outliers(df, col):
    print(f"{col} col have outliers")
  else:
    print(f"{col} doesn't have outliers")

def check_for_NA(dataframe, variable):
  return dataframe[variable].isnull().values.any()

for col in num_cols:
  if check_for_NA(df, col):
    print(f"{col} col have missing values")
  else:
    print(f"{col} doesn't have missing values")


#### deleting outliers

def get_outlier_limits(dataframe, col):
  q1 = dataframe[col].quantile(0.25)
  q3 = dataframe[col].quantile(0.75)
  iqr = q3 - q1
  low_limit = q1 - 1.5 * iqr
  up_limit = q3 + 1.5 * iqr
  return low_limit, up_limit


def delete_outliers(dataframe, col):
  low_limit, up_limit = get_outlier_limits(dataframe, col) 
  df_without_outliers = dataframe[~((dataframe[col] < low_limit)|(dataframe[col] > up_limit))]
  return df_without_outliers

def replace_outliers(dataframe, col):
  low_limit, up_limit = get_outlier_limits(dataframe, col) 
  dataframe[dataframe[col] < low_limit] = low_limit
  dataframe.loc[(dataframe[col] >  up_limit), col] = up_limit
  dataframe.loc[(dataframe[col] <  low_limit), col] = low_limit



for col in num_cols:
  df = delete_outliers(df, col)


for col in num_cols:
  if check_for_outliers(df, col):
    print(f"{col} col have outliers")
  else:
    for col in num_cols:
      replace_outliers(df, col)

for col in num_cols:
  if check_for_outliers(df, col):
    print(f"{col} col have outliers")
  else:
    print(f"{col} doesn't have outliers")


#### correlation analysis

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

#### Missing values

df.isnull().sum().sort_values(ascending=False) #Shows no missing values

df.loc[(df["Insulin"] == 0)].head(5) 
# Insulin must be bigger than 0 this means NA values have filled with 0s in this col

df.loc[(df["Glucose"] == 0) ].head(5)

df.loc[(df["DiabetesPedigreeFunction"] == 0)].head(3)

### Replacing 0s to Nan
insulin_zero_index_list = df.index[df["Insulin"] == 0].to_list()

df.loc[insulin_zero_index_list, "Insulin"] = np.nan

df.loc[(df["Insulin"] == 0)].head(5) 

### functional version

def check_for_zeros(dataframe, col):
  if (dataframe[col] == 0).any():
    return True
  else:
    return False
  
for col in num_cols:
  if check_for_zeros(df,col):
    print(f"{col} have unnecessary 0s")
  else:
    print(f"{col} filled normally")

def replace_zeros(dataframe, col):
  zero_index_list = dataframe.index[df[col] == 0].to_list()
  dataframe.loc[zero_index_list, col] = np.nan

for col in num_cols:
  replace_zeros(df, col)