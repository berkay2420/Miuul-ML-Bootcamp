import random
import catboost
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axis, pyplot as plt
import missingno as msno
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sympy import lowergamma
from tables import Cols
import test



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
df.isnull().sum()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"] , errors='coerce')

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
num_cols.append("TotalCharges")



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

#df["Churn"] = df["Churn"].map({'Yes':1, 'No':0})

df["Churn"].head(50)

df.groupby("PaymentMethod")["Churn"].mean()

df.groupby("gender")["Churn"].mean()

def target_summary_with_cat(dataframe, col, target):
  print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(col)[target].mean(),
                      "COUNT": dataframe[col].value_counts(),
                      "Ratio": 100 * dataframe[col].value_counts() / len(dataframe)}), end="\n\n\n")

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
  if df[(df[col] < low)  | (df[col] > up)].any(axis=None):
    return True
  else:
    return False

num_cols
check_for_outliers(df, "tenure")
check_for_outliers(df, "MonthlyCharges")  
check_for_outliers(df,"TotalCharges")
### missing values analysis




df["TotalCharges"].isnull().sum()

df["TotalCharges"].isnull(axis=1).count()

msno.bar(df) 
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

def missing_values_table(dataframe):
  na_cols = [col for col in dataframe.columns if dataframe[col].isnull().values.any()]

  for col in na_cols:
    print(f" Pertence of NA values in {col}:{(dataframe[col].isnull().sum() / dataframe.shape[0] * 100)}")

missing_values_table(df)

def advanced_missing_values_table(dataframe, na_name=False):
  na_cols = [col  for col in dataframe.columns if dataframe[col].isnull().values.any()]
  n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending= False)
  ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
  missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=['n_miss','ratio'])
  print(missing_df, end="\n")

  if na_name:
    return na_cols

advanced_missing_values_table(df)

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df.isnull().sum()
### correlation analysis 

df[num_cols].corr()
f, ax = plt.subplots(figsize=[18,13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()
#this graph shows there is 83% correlation between tenure and total charges


df.corrwith(df["Churn"]).sort_values()


df.head(3)
df.describe()
df.dtypes

### base model
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["Churn"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
  dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
  return dataframe

dff = one_hot_encoder(dff, cat_cols, drop_first=True )

y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from catboost import CatBoostClassifier
cat_boost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = cat_boost_model.predict(X_test)

print(f"Accuracy:{round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall:{round(recall_score(y_pred, y_test), 4)}")
print(f"Precision:{round(precision_score(y_pred, y_test), 4)}")
print(f"F1 Score:{round(f1_score(y_pred, y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test),4 )}")

#CatBoostClassifier Handles categorical variables automatically thats why we used
#in "base" model



#### new variable creations

df["tenure"].max()


df.loc[(df["tenure"] >=0) & (df["tenure"]<=12), "NEW_TENURE_YEAR"] = "0-1_YEAR"
df.loc[(df["tenure"] >12) & (df["tenure"]<=24), "NEW_TENURE_YEAR"] = "1-2_YEAR"
df.loc[(df["tenure"] >24) & (df["tenure"]<=36), "NEW_TENURE_YEAR"] = "2-3_YEAR"
df.loc[(df["tenure"] >36) & (df["tenure"]<=48), "NEW_TENURE_YEAR"] = "3-4_YEAR"
df.loc[(df["tenure"] >48) & (df["tenure"]<=60), "NEW_TENURE_YEAR"] = "4-5_YEAR"
df.loc[(df["tenure"] >60) & (df["tenure"] <=72), "NEW_TENURE_YEAR"] = "5-6_YEAR"

cat_cols

df["NEW_ENGAGED"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)
 
df["NEW_Young_Not_Enagaged"] = df.apply(lambda x: 1 if (x["NEW_ENGAGED"] == 0) and (x["SeniorCitizen"] == 0) else 0, axis=1 )
# axis=1 means work on rows

# df["NEW_NO_INTERET"] = df.apply(lambda x: 1 if (x["OnlineSecurity"] == "No internet service") or (x["OnlineBackup"] == "No internet service")
#                                 or (x["DeviceProtection"] == "No internet service") or (x["TechSupport"] == "No internet service") or 
#                                 (x["StreamingTV"] == "No internet service") or (x["StreamingMovies"] == "No internet service") else 0)

def check_for_no_internet(dataframe,categorical_cols):
  no_internet_cols = []
  for col in categorical_cols:
    if (dataframe[col] == "No internet service").any():
        no_internet_cols.append(col)
  return no_internet_cols

no_internet_cols = check_for_no_internet(df, cat_cols)

df["NEW_NO_INTERNET"] = (df[no_internet_cols] == "No internet service").any(axis=1).astype(int)


df["NEW_MONTHLY_BELOW_AVERAGE"] = df["MonthlyCharges"].apply(lambda x: 1 if x > df["MonthlyCharges"].mean() else 0 )

df["NEW_TOTAL_BELOW_AVERAGE"] = df["TotalCharges"].apply(lambda x: 1 if x > df["TotalCharges"].mean() else 0 )

df["NEW_TOTAL_SERVICES"] = (df[["PhoneService", "InternetService",
                               "OnlineSecurity", "OnlineBackup",
                               "DeviceProtection", "TechSupport",
                               "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)



df["NEW_FLAG_AUTO_PAYMENT"] = df["PaymentMethod"].apply(lambda x: 1 if x in  ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)


df["NEW_AVG_Charges"] = df["TotalCharges"] / df["tenure"] + 1

df["NEW_INCREASE"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

df["NEW_AVG_SERVICE_FEE"] = df["MonthlyCharges"] / (df["NEW_TOTAL_SERVICES"] + 1)


df.head()


#### Label Encoding
cat_cols , num_cols, cat_but_car = grab_cols(df)

def label_encoder(dataframe, binary_col):
  label_encoder = LabelEncoder()
  dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
  return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and  df[col].nunique() == 2]

for col in binary_cols:
  df = label_encoder(df, col)

df.head()

### One hot encoding
cat_cols =  [col for col in cat_cols if col not in ["Churn", "NEW_TOTAL_SERVICES"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
  dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
  return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

### main model

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from catboost import CatBoostClassifier
cat_boost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = cat_boost_model.predict(X_test)

print(f"Accuracy:{round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall:{round(recall_score(y_pred, y_test), 4)}")
print(f"Precision:{round(precision_score(y_pred, y_test), 4)}")
print(f"F1 Score:{round(f1_score(y_pred, y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test),4 )}")



###
def plot_feature_importance(importance, names, model_type):
  feature_importance = np.array(importance)
  feature_names = np.array(names)

  data = {'feature_names': feature_names, 'feature_importance': feature_importance}
  f1_df = pd.DataFrame(data)

  f1_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

  plt.figure(figsize=(15,10))
  sns.barplot(x=f1_df['feature_importance'], y=f1_df['feature_names'])

  plt.title(model_type + 'FEATURE_IMPORTANCE')
  plt.xlabel('FEATURE_IMPORTANCE')
  plt.ylabel('FETURE_NAMES')
  plt.show()

plot_feature_importance(cat_boost_model.get_feature_importance(), X.columns, 'CATBOOST')