from itertools import dropwhile, groupby
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
from sklearn.model_selection import learning_curve, train_test_split
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

###########################
##### Missing Values ######
###########################

#### Catching Missing Values ####
df = load()
df.head()

df.isnull().values.any()

df.isnull().sum()    

df.notnull().sum()

df.isnull().sum().sum()  

df[df.isnull().any(axis=1)] #Displaying rows with null cells

df[df.notnull().any(axis=1)]

df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col  for col in df.columns if df[col].isnull().values.any()]

###
def missin_values_table(dataframe):
  na_cols = [col  for col in dataframe.columns if dataframe[col].isnull().values.any()]

  for col in na_cols:

    print(f" Pertence of NA values in {col}:{(dataframe[col].isnull().sum() / dataframe.shape[0] * 100)}")
    print("###################################")

missin_values_table(df)

def missing_values_table(dataframe, na_name=False):
  na_cols = [col  for col in dataframe.columns if dataframe[col].isnull().values.any()]
  n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending= False)
  ratio = (dataframe[na_cols].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
  missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=['n_miss','ratio'])
  print(missing_df, end="\n")

  if na_name:
    return na_cols
  
missing_values_table(df)

missing_values_table(df, na_name=True)

df.dropna()

df["Age"].fillna(df["Age"].mean())

df.aplly(lambda x: x.fillna(x.mean()), axis=0) #returns erorr

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'O' else x, axis=0).head()

dff= df.apply(lambda x: x.fillna(x.mean()) if x.dtype != 'O' else x, axis=0)

df = load()
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == 'O' and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

####

df.groupby("Sex")["Age"].mean() 

df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Filling NA values with mean of their category. 
# (female age mean and male age mean are different)

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female")].apply(
                                                              lambda x: x.fillna(df.groupby("Sex")["Age"].transform("mean"))
                                                              ).head(5)

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]


#####

df = load()

cat_cols, num_cols, cat_but_car = grab_cols(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
# Convert categorical variable into dummy/indicator variables.
# Most machine learning models doesn't accept categorica data. These datas needs to be converted into
# numerical data. For this we use "One-Hot Encoding". One hot encoding is a technique that we use to represent 
# categorical variables as numerical values in a machine learning model.
# If there is Male and Female categories, one hot encoding gives 0 and 1. But this way one is superior to the other.
# For solving this problem one hot makes 2 more columns male and female. Gives 0 to male while female is 1 and gives 
# 0 to female while male is 1.

dff.head(5)

dff = pd.get_dummies(df[cat_cols + num_cols], columns=['Sex'])
# 2 more columns "Sex_female"	"Sex_male" created.
dff.head(5)

#### Variable Standartization ####
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

#### kNN Implementation 

# Filling missing value with machine learning
# Model tries to fill values

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head(10)

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
# unscale values

df["knn_imputed_age"] = dff["Age"]

df.loc[df["Age"].isnull(), ["Age", "knn_imputed_age"]]

df.head(5)

## Recap ##
# -Sclaed the data using MinMaxScaler and created new data frame (dff)
# -With using knn imputer filled NA values with machine learning
# -Then created "knn_imputed_age" column using "Age" col from dff
# -Now age and "knn_imputed_age" can be shown at the same time
# -This means machine filled NA values 

### Examining Missing Data ####

msno.bar(df) #showing integer count in each col
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

#############################

missing_values_table(df)
na_cols = missing_values_table(df, na_name= True)

### Trying to find cause of survival rate

def missing_vs_target(dataframe, target, na_columns):
  temp_df = dataframe.copy()
  for col in na_columns:
    temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(),1,0) 
    # returns 0 or 1 if col have NA values 
  na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
  for col in na_flags:
    print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                        "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
    

missing_vs_target(df, "Survived", na_cols)