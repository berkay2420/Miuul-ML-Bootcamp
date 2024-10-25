import pandas as pd
import numpy as np
from requests import get
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Log, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate, learning_curve, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sympy import root
import test

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
df.isna().sum()
df.isnull().sum()
(df == 0).sum()

df["Salary"] = df["Salary"] * 1000

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
  target_summary_with_numerical(df, col, "Salary")

##NA & Missing Values Analysis

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
  dataframe.loc[dataframe[col] > up, col ] = up
  dataframe.loc[dataframe[col] < low, col] = low

for col in num_cols:
  replace_outliers(df, col)

for col in num_cols:
  print(col, check_for_outliers(df,col))


df.isnull().sum() 


## Correlation Analysis ##
df[num_cols].corr()
f, ax = plt.subplots(figsize=[18,13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

### Base Model ###
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
  dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
  return dataframe

cat_cols, num_cols, cat_but_car = grab_cols(df)

df = one_hot_encoder(df, categorical_cols=cat_cols, drop_first=True)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Log, LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate, learning_curve, train_test_split

num_cols = [col for col in num_cols if col not in ["Salary"]]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()


y = df["Salary"] 
X= df.drop("Salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=47)

lin_reg_model = LinearRegression().fit(X_train, y_train)
y_pred = lin_reg_model.predict(X_test)

print("mean squared error",mean_squared_error(y_pred, y_test))
print("root mean squared error",mean_squared_error(y_pred, y_test,  squared=False))
print("R2 SCORE", r2_score(y_test, y_pred))

# mean squared error 67736.38372146642
# root mean squared error 260.26214423435925
# R2 SCORE 0.48478760306166413


### Feature Engineering ###

num_cols = [col for col in num_cols if col not in ["Salary"]]
df[num_cols] = df[num_cols]+0.0000000001

df["NEW_CHIT_ACCURACY_RATIO"] = 100 * df["CHits"] / df["CAtBat"]

df["NEW_HIT_ACCURACY_RATIO_86"] = 100 * df["Hits"] / df["AtBat"]

df["NEW_HITS"]  =df["Hits"] / df["CHits"] + df["Hits"]
df["NEW_CRBI*CATBAT"] = df["CRBI"] * df["CAtBat"]

df["NEW_Chits"] = df["CHits"] / df["Years"]
df["NEW_CHmRun"] = df["CHmRun"] / df["Years"]
df["NEW_CRuns"] = df["CRuns"] / df["Years"]

df["NEW_DIFF_AtBat"] = df["AtBat"] / (df["CAtBat"] / df["Years"])
df["NEW_DIFF_Hits"] = df["Hits"] / (df["CHits"] / df["Years"])
df["NEW_DIFF_HmRun"] = df["HmRun"] / (df["CHmRun"] / df["Years"])
df["NEW_DIFF_Runs"] = df["Runs"] / (df["CRuns"] / df["Years"])
df["NEW_DIFF_RBI"] = df["RBI"] / (df["CRBI"] / df["Years"])
df["NEW_DIFF_Walks"] = df["Walks"] / (df["CWalks"] / df["Years"])

df.columns

cat_cols, num_cols, cat_but_car = grab_cols(df)
## encoding 

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
  dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=True)
  return dataframe

cat_cols, num_cols, cat_but_car = grab_cols(df)

df = one_hot_encoder(df, categorical_cols=cat_cols, drop_first=True)
df.head(3)
df.isnull().sum()

## scaling 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Log, LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, learning_curve, train_test_split

num_cols = [col for col in num_cols if col not in ["Salary"]]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

### modelig
df["Salary"] = df["Salary"]  / 1000
y = df["Salary"]

X = df.drop(["Salary"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=47)

lin_reg_model = LinearRegression().fit(X_train, y_train)

y_pred = lin_reg_model.predict(X_test)

print("mean squared error",mean_squared_error(y_pred, y_test))
print("root mean squared error",mean_squared_error(y_pred, y_test,  squared=False))
print("R2 SCORE", r2_score(y_test, y_pred))

# mean squared error 5.985661085397567e-08
# root mean squared error 0.0002446561073302191
# R2 SCORE 0.5447222562472164

