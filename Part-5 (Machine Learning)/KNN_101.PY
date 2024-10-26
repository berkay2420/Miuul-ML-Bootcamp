import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.linear_model import Log, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, learning_curve, train_test_split
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler

######################################
##### KNN With Diabetes DataSet ######
######################################

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' %x)
pd.set_option('display.width',500)

df = pd.read_csv("diabetes.csv")
df.head(3)


df.describe
df.dtypes
df.shape
df["Outcome"].value_counts()


######
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_scaled = StandardScaler().fit_transform(X) #without column names

X = pd.DataFrame(X_scaled, columns=X.columns) # column names added

### Modeling

knn_model = KNeighborsClassifier().fit(X,y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)


### Model Evaluation

y_pred = knn_model.predict(X)

y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))

roc_auc_score(y, y_prob)

#Cross Validation

cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

knn_model.get_params()

#######################################
##### Hyperparameter Optimization #####
#######################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {
  "n_neighbors": range(2,50), 
}

knn_gs_best = GridSearchCV(knn_model, knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X,y)

knn_gs_best.best_params_

#### Final Model ####

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final, X, y, cv=5, scoring=["accuracy","f1","roc_auc"])

cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()