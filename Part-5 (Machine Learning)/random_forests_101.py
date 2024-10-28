from re import I
import warnings
import numpy as np
import pandas as pd
import seaborn  as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("diabetes.csv")
df.head(3)

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

##########################
####  RANDOM FORESTS  ####
##########################

rf_model = RandomForestClassifier(random_state=17)

rf_model.get_params()

rf_params = {"max_depth":[5,8,None],
             "max_features": [3,5,7,"auto"],
             "min_samples_split":[2,5,8,15,20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=10,
                            n_jobs=-1,
                            verbose=True).fit(X,y)

print(rf_best_grid.best_params_)

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X,y)

cv_results =  cross_validate(rf_final,X,y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()