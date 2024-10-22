from re import S
from tarfile import data_filter
import pandas as pd
import numpy as np
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

df = pd.read_csv("diabetes.csv")
df.head(3)


def get_outlier_tresholds(dataframe, col, q1=0.05, q3=0.95):
  q1 = dataframe[col].quantile(q1)
  q3 = dataframe[col].quantile(q3)
  iqr = q3 - q1
  low_limit = q1 - 1.5 * iqr
  up_limit = q3 + 1.5 * iqr
  return low_limit, up_limit

def check_for_outliers(dataframe, col):
  low, up = get_outlier_tresholds(dataframe, col)
  if dataframe[(dataframe[col] < low)  | (dataframe[col] > up)].any(axis=None):
    return True
  else:
    return False
  
def replace_with_tresholds(dataframe, variable):
  low, up = get_outlier_tresholds(dataframe, variable)
  dataframe.loc[dataframe[variable] < low, variable] = low
  dataframe.loc[dataframe[variable] > up, variable] = up


### Exploratory Data Analysis ###
df.shape

### Target Analysis
df["Outcome"].value_counts()
sns.countplot(x="Outcome", data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

### Feature Analysis
df.describe().T #only numerical features

# visualization for numrical = hist, boxplot

df["BloodPressure"].hist(bins=20)
plt.xlabel("Glcose")
plt.show()

def plot_numrical_cols(dataframe, numerical_col):
  dataframe[numerical_col].hist(bins=20)
  plt.xlabel(numerical_col)
  plt.show(block=True)

for col in df.columns:
  plot_numrical_cols(df, col)

cols = [col for col in df.columns if "Outcome" not in col]

### Target vs Features

def target_vs_features(dataframe, target, feature):
  print(dataframe.groupby(target).agg({feature:"mean"}))

for col in cols:
  target_vs_features(df, "Outcome", col)

df.groupby("Outcome").agg({"Pregnancies":"mean"})

### Data Preprocessing
df.isnull().sum()

for col in cols:
  print(col, check_for_outliers(df,col))

replace_with_tresholds(df,"Insulin")

### Scaling
for col in df.columns:
  df[col] = RobustScaler().fit_transform(df[[col]]) 
  # Robuts perfoms better for outliers
df.head()

#### Model & Prediction ####
y = df["Outcome"]
X= df.drop(["Outcome"], axis=1)

log_reg_model = LogisticRegression().fit(X,y)

#bias
log_reg_model.intercept_

#weighrs
log_reg_model.coef_

y_pred = log_reg_model.predict(X)

y_pred[0:10]

y[0:10]


#### Model Evaluation ####
def plot_confuison_matrix(y, y_pred):
  acc = round(accuracy_score(y, y_pred), 2)
  cm = confusion_matrix(y, y_pred)
  sns.heatmap(cm, annot=True, fmt=".0f")
  plt.xlabel("y_pred")
  plt.ylabel("y")
  plt.title(f"Accuracy Score {acc}", size=10)
  plt.show()


plot_confuison_matrix(y, y_pred)

print(classification_report(y, y_pred))

#ROC-AUC
y_prob = log_reg_model.predict_proba(X)[:,1]
roc_auc_score(y, y_pred)


#### Model Validation: Holdout ####

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=17)


log_reg_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_reg_model.predict(X_test) 

y_prob = log_reg_model.predict_log_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))

#### Model Validation: 10-Fold Cross Validation ####

log_reg_model = LogisticRegression().fit(X,y)

cv_results = cross_validate(log_reg_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy","precision","recall","f1","roc_auc"])

cv_results['test_accuracy'].mean()

cv_results['test_precision'].mean()


cv_results['test_recall'].mean()


cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

#### Prediction for new observation

random_user = X.sample(1, random_state=45)

log_reg_model.predict(random_user)
# array([1.]) model predicted as diabetes