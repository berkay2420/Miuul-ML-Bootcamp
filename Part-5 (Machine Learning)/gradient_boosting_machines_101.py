from tabnanny import verbose
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

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params() 
#n_estimators burada optimizasyon sayısı ağaç değil

cv_results = cross_validate(gbm_model, X, y, cv=10, scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3,8,10],
              "n_estimators":[100,500,1000],
              "subsample":[1,0.5,0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

print(gbm_best_grid.best_params_)

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X,y)

cv_results = cross_validate(gbm_final, X, y, cv=10, scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

#################
###  XGBOOST  ###
#################

xgboost_model = XGBClassifier(random_state=17)

cv_results = cross_validate(xgboost_model, X, y, cv=10, scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
#0.7434723171565276
cv_results["test_f1"].mean()
#0.622106290128009
cv_results["test_roc_auc"].mean()
#0.7921794871794872

xgboost_params = {"learning_rate":[0.1,0.01,0.001],
                  "max_depth":[5,8,12,15,20],
                  "n_estimators":[100,500,1000],
                  "colsample_bytree":[0.5,0.7,1]}

xgboost_best_grid = GridSearchCV(xgboost_model,xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

print(xgboost_best_grid.best_params_)

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X,y)

cv_results = cross_validate(xgboost_final, X, y, cv=10, scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
#0.7551948051948052
cv_results["test_f1"].mean()
#0.6314120140189479
cv_results["test_roc_auc"].mean()
#0.8178005698005698

#################
### LightGBM ####
#################

lgbm_model = LGBMClassifier(random_state=17)

cv_results = cross_validate(lgbm_model, X, y, cv=10, scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
#0.7409432672590568
cv_results["test_f1"].mean()
#0.6075187280502355
cv_results["test_roc_auc"].mean()
#0.8054216524216524

print(lgbm_model.get_params())


lgbm_params = {
    'num_leaves': [5, 20, 31],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 150]
}

lgbm_best_grid = GridSearchCV(lgbm_model,lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

print(lgbm_best_grid.best_params_)

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X,y)

cv_results = cross_validate(lgbm_final, X, y, cv=10, scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
#0.7604408749145591
cv_results["test_f1"].mean()
#0.6113294314381271
cv_results["test_roc_auc"].mean()
#0.8276837606837606



#################
### CatBoost ####
#################

catboost_model = CatBoostClassifier(random_state=17,verbose=False)

cv_results = cross_validate(catboost_model, X, y, cv=10, scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()
#0.7656015037593985
cv_results["test_f1"].mean()
#0.6407485394029232
cv_results["test_roc_auc"].mean()
#0.8344045584045585

print(catboost_model.get_params())

catboost_params = {
    "iterations":[200,500],
    "learning_rate":[0.01,0.1],
    "depth":[3,6]
}

catboost_best_grid = GridSearchCV(catboost_model,catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X,y)

print(catboost_best_grid.best_params_)

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params, random_state=17).fit(X,y)

cv_results = cross_validate(catboost_final, X, y, cv=10, scoring=["accuracy","f1","roc_auc"])
cv_results["test_accuracy"].mean()

cv_results["test_f1"].mean()

cv_results["test_roc_auc"].mean()
