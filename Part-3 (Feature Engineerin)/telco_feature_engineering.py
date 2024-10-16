import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axis, pyplot as plt
import missingno as msno
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sympy import lowergamma



pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' %x)
pd.set_option('display.width',500)

df = pd.read_csv("Telco-Customer-Churn.csv")
df.head(5)
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

### categorical col analysis
def cat_summary(dataframe,col, plot=False):
  if dataframe[col].dtypes == "bool": 
    dataframe[col] = dataframe[col].astype(int)

  print(pd.DataFrame({col: dataframe[col].value_counts(), 
                      "Ratio": 100*dataframe[col].value_counts()/len(dataframe)}))
  
  if plot:
    sns.countplot(x=dataframe[col], data=dataframe)
    plt.show(block=True)    
 

for col in cat_cols:
 cat_summary(df, col)

for col in cat_cols:
  print(df[col].value_counts())
### numerical col analysis
def num_summary(dataframe, numerical_col):
  quantiles = [0.05, 0.1, 0.2 ,0.3, 0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,0.99]
  print(dataframe[numerical_col].describe(quantiles).T)

for col in num_cols:
  num_summary(df, col)

### target analysis for categoricals

cat_cols

df["Churn"] = df["Churn"].map({'Yes':1, 'No':0})

df["Churn"].head(50)

df.groupby("PaymentMethod")["Churn"].mean()

df.groupby("gender")["Churn"].mean()

def target_summary_with_cat(dataframe, col, target):
  print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(col)[target].mean()}))

target_summary_with_cat(df, "Contract", "Churn")

for col in cat_cols:
  target_summary_with_cat(df, col, "Churn")

### target analysis for numericals

num_cols

def target_summary_with_num(dataframe, col, target):
  print(dataframe.groupby(target).agg({col:"mean"}), end="\n\n\n")

for col in num_cols:
  target_summary_with_num(df, col, "Churn")

### Outlier analysis

def get_outlier_tresholds(dataframe, col):
  q1 = dataframe[col].quantile(0.25)
  q3 = dataframe[col].quantile(0.75)
  iqr = q3 - q1
  low_limit = q1 - 1.5 * iqr
  up_limit = q3 + 1.5 * iqr
  return low_limit, up_limit

def check_for_outliers(dataframe, col):
  low, up = get_outlier_tresholds(dataframe, col)
  if df[(df[col] < low) & (df[col] > up)].any(axis=None):
    return True
  else:
    return False

num_cols
check_for_outliers(df, "tenure")
check_for_outliers(df, "MonthlyCharges")

### missing values analysis

df.isnull().values.any()

### correlation analysis 