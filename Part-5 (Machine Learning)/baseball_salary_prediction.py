import pandas as pd
import numpy as np
from requests import get
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Log, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate, learning_curve, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler

########################################################
##### Diabete Prediction with Logistic Regression ######
########################################################

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' %x)
pd.set_option('display.width',500)

df = pd.read_csv("hitters.csv")
df.head(3)

df.info
df.dtypes
df.shape
df.isnull().any()
df["Salary"].head()
df.count()

df.describe().T

df.isnull().values.any()

### Data Preprocessing ###

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
  cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
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
  print(f"numerical_but_categorical:{len(num_but_cat )}")

  return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_cols(df)


df.groupby("League").agg({"Salary":"mean"})

def target_analysis(dataframe, col, target):
  print("Mean Salary")
  print(dataframe.groupby(col).agg({target: "mean"}))

for col in df.columns:
  target_analysis(df, col, "Salary")

def categorical_col_summary(dataframe, categorical_col):
  print(pd.DataFrame({categorical_col : dataframe[categorical_col].value_counts(),
                      "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe) }))

cat_cols, num_cols

for col in cat_cols:
  categorical_col_summary(df, col)

for col in cat_cols:
  print(df[col].value_counts())
  
#target analysis for categorical cols
def target_summary_with_cat(dataframe, categorical_col, target):
  print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                      "COUNT" : dataframe[categorical_col].value_counts(),
                      "RATIO": 100 * dataframe[categorical_col].value_counts() / len(dataframe) }))
  

for col in cat_cols:
  target_summary_with_cat(df, col, "Salary")

#target analysis for numrical cols
def target_summary_with_numerical(dataframe, numerical_col, target):
  print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(numerical_col)[target].mean()}), end="\n\n\n")

for col in num_cols:
  target_summary_with_cat(df, col, "Salary")

### Outlier Analysis

def get_outlier_tresholds(dataframe, col, q1=0.25, q3=0.75):
  q1 = dataframe[col].quantile(q1)
  q3 = dataframe[col].quantile(q3)
  iqr = q3 - q1 
  low_limit = q1 - 1.5 * iqr
  up_limit  = q3 + 1.5 * iqr
  return low_limit, up_limit

def check_for_outliers(dataframe, col):
  low, up = get_outlier_tresholds(dataframe, col)
  if dataframe.loc[(dataframe[col] < low) | (dataframe[col] > up)].any(axis=None):
    return True
  else:
    return False

for col in num_cols:
  print(col, check_for_outliers(df,col))

def replace_outliers(dataframe, col):
  low, up = get_outlier_tresholds(dataframe, col)
  dataframe[dataframe[col] > up ] = up
  dataframe[dataframe[col] < low] = low

for col in num_cols:
  replace_outliers(df, col)

for col in num_cols:
  print(col, check_for_outliers(df,col))


df.isnull().sum() 

def advanced_missing_values_table(dataframe, na_name=False):
  na_cols = [col  for col in dataframe.columns if dataframe[col].isnull().values.any()]
  n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending= False)
  ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
  missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=['n_miss','ratio'])
  print(missing_df, end="\n")

  if na_name:
    return na_cols
  
advanced_missing_values_table(df)

df["Salary"].fillna(df["Salary"].median(), inplace=True)

df.isnull().sum()