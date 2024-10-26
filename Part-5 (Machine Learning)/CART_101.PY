import warnings
from catboost import train
import pandas as pd
import numpy as np
import joblib
import pydotplus
import astor
import seaborn as sns
from matplotlib import category, pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile

pd.set_option('display.max_columns', None)
warnings.simplefilter(action="ignore", category=Warning)

### Modeling with CART ###

df=pd.read_csv("diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# y_pred for confuison matrix 
y_pred = cart_model.predict(X)

# y_prob for roc_auc_score
y_prob = cart_model.predict_proba(X)[:, 1]

# Confusion Matrix
print(classification_report(y, y_pred))

# AUC
roc_auc_score(y, y_prob)

### Evaluation with Hold Out ###
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=17,
                                                    test_size=0.30)

cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# train 
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:,1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

### Cross Validation 

cart_model = DecisionTreeClassifier(random_state=17).fit(X,y)

cv_results = cross_validate(cart_model, X, y,
                            cv=5,
                            scoring=["accuracy","f1","roc_auc"])

print("Mean Accuracy:", np.mean(cv_results['test_accuracy']))
print("Mean F1 Score:", np.mean(cv_results['test_f1']))
print("Mean ROC AUC:", np.mean(cv_results['test_roc_auc']))

# Mean Accuracy: 0.7058568882098294
# Mean F1 Score: 0.5710621194523633
# Mean ROC AUC: 0.6719440950384347

### Hyper Parameter Optimization ###

cart_model.get_params()

cart_params = {
  'max_depth':range(1,11),
  'min_samples_split': range(2,20)
  }

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              cv=5,
                              n_jobs=-1, #Using CPUs in full performence 
                              verbose=True).fit(X,y)
#grid searchCV gives best parameters for accuracy by default.
# Changing to  scoring="f1" would give best parameters for f1 score  



cart_best_grid.best_params_

cart_best_grid.best_score_ 

random = X.sample(1, random_state=45)

cart_best_grid.predict(random) #using grid searchCV as a model 

### Final Model ###

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X,y)

cart_final.get_params()

### Another way for final model
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)

cv_results = cross_validate(cart_final, X, y,
                            cv=5,
                            scoring=["accuracy","f1","roc_auc"])

print("Mean Accuracy:", np.mean(cv_results['test_accuracy']))
print("Mean F1 Score:", np.mean(cv_results['test_f1']))
print("Mean ROC AUC:", np.mean(cv_results['test_roc_auc'])) 

# Mean Accuracy: 0.7500806383159324
# Mean F1 Score: 0.614625004082526
# Mean ROC AUC: 0.797796645702306
