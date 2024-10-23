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
  target_summary_with_numerical(df, col, "Salary")

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

## Correlation Analysis ##
df[num_cols].corr()
f, ax = plt.subplots(figsize=[18,13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

### Feature Extraction ###

df["NEW_HIT_ACCURACY_RATIO"] = 100 * df["CHits"] / df["CAtBat"]

df.loc[df["NEW_HIT_ACCURACY_RATIO"] < 11, "NEW_HIT_ACCURACY_RATIO_CLASSIFICATIN"] = "0_10 Pertence"
df.loc[(df["NEW_HIT_ACCURACY_RATIO"] >11)&(df["NEW_HIT_ACCURACY_RATIO"] < 21), "NEW_HIT_ACCURACY_RATIO_CLASSIFICATIN"] = "10_20 Pertence"
df.loc[(df["NEW_HIT_ACCURACY_RATIO"] >20)&(df["NEW_HIT_ACCURACY_RATIO"] < 31), "NEW_HIT_ACCURACY_RATIO_CLASSIFICATIN"] = "20_30 Pertence"



df["NEW_HIT_ACCURACY_RATIO_86-87"] = 100 * df["Hits"] / df["AtBat"]

df["NEW_HIT_ACCURACY_RATIO_86-87"].value_counts()

df["NEW_SCORE_CONT_PER_YEAR"] =   df["CRuns"] / df["Years"]


df["NEW_SCORE_CONT_PER_YEAR"].value_counts()

df["HmRun"].value_counts()

df.loc[df["HmRun"] <= 5, "NEW_HM_RUN"] = "0_5"
df.loc[(df["HmRun"] > 5) & (df["HmRun"] <= 15), "NEW_HM_RUN"] = "10_15"
df.loc[(df["HmRun"] > 15) & (df["HmRun"] <= 20), "NEW_HM_RUN"] = "15_20"
df.loc[(df["HmRun"] > 20) & (df["HmRun"] <= 25), "NEW_HM_RUN"] = "20_25"
df.loc[(df["HmRun"] > 25) & (df["HmRun"] <= 30), "NEW_HM_RUN"] = "25_30"
df.loc[(df["HmRun"] > 30) & (df["HmRun"] <= 35), "NEW_HM_RUN"] = "30_35"


df["NEW_WALKS"] = df["Walks"] / df["CWalks"]
df["NEW_AT_BAT"] =  df["AtBat"] / df["CAtBat"]
df['NEW_HITS'] = df['Hits'] / df['CHits'] + df['Hits']
df['NEW_RBI'] = df['RBI'] / df['CRBI']

df.loc[df["NEW_SCORE_CONT_PER_YEAR"] > df["Runs"], "NEW_SCORE_PERFOMANCE"] = "above"
df.loc[df["NEW_SCORE_CONT_PER_YEAR"] < df["Runs"], "NEW_SCORE_PERFOMANCE"] = "below"


# Replace infinite values with NaN, then handle missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check for NaNs
print(df.isnull().sum())

# Fill or remove any remaining NaNs
df.fillna(df.median(), inplace=True)

## encoding 

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
  dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=True)
  return dataframe

cat_cols, num_cols, cat_but_car = grab_cols(df)

df = one_hot_encoder(df, categorical_cols=cat_cols, drop_first=True)
df.head(3)
df.isnull().sum()

df["NEW_RBI"].fillna(df["NEW_RBI"].median(), inplace=True)
## scaling 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Log, LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate, learning_curve, train_test_split

num_cols = [col for col in num_cols if col not in ["Salary"]]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()

### modelig
y = df["Salary"]

X = df.drop(["Salary"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=47)

lin_reg_model = LinearRegression().fit(X_train, y_train)

y_pred = lin_reg_model.predict(X_test)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))


from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, y_pred))

from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))


print("root_mean_squared_error:", np.sqrt(mean_absolute_error(y_test, y_pred)))