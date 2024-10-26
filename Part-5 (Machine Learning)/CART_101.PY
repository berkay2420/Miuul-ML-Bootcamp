import warnings
from catboost import train
from cv2 import ScoreMethod
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

#### Feature Importance ####
cart_final.feature_importances_

def plot_importance(model, features, num=len(X), save=False):
  feature_imp = pd.DataFrame({'Values': model.feature_importances_,
                              'Feature': features.columns})
  plt.figure(figsize=(10,10))
  sns.set(font_scale=1)
  sns.barplot(x="Values", y="Feature", data=feature_imp.sort_values(by="Values",
                                                                  ascending=False)[0:num])
  
  plt.title("Features")
  plt.tight_layout()
  plt.show()
  if save:
    plt.savefig('importances.png')

plot_importance(cart_final, X)

#######################################################
#### Analyzing Model Complexity with Lerning Curves ###
#######################################################

train_score, test_score = validation_curve(cart_final, X, y, 
                                           param_name="max_depth",
                                           param_range=range(1,11),
                                           scoring="roc_auc",
                                           cv=10)

mean_train_score = np.mean(train_score, axis=1)
mean_test_score = np.mean(test_score, axis=1)
print(mean_train_score)
print(mean_test_score)

plt.plot(range(1,11), mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1,11), mean_test_score,
         label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of Max Depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc="best")
plt.show()

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
  train_score, test_score = validation_curve(model, X, y,
                                             param_range=param_range,
                                             param_name=param_name,
                                             scoring=scoring,
                                             cv=cv)
  mean_train_score = np.mean(train_score, axis=1)
  mean_test_score = np.mean(test_score, axis=1)

  plt.plot(param_range, mean_train_score,
         label="Training Score", color='b')

  plt.plot(param_range, mean_test_score,
          label="Validation Score", color='g')

  plt.title(f"Validation Curve for {type(model).__name__}")
  plt.xlabel(f"Number of {param_name}")
  plt.ylabel(f"{scoring}")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.show()

val_curve_params(cart_final, X, y, "max_depth", range(1,11),scoring="f1")

val_curve_params(cart_final, X, y, "max_depth", range(1,11),scoring="accuracy")

cart_val_params = [["max_depth",range(1,11)], ["min_samples_split", range(2,20)]]

for i in range(len(cart_val_params)):
  val_curve_params(cart_final, X, y, cart_val_params[i][0], cart_val_params[i][1])


#######################################
#### Visualizing the Decision Tree ####
#######################################

import graphviz 

def tree_graph(model, col_names, file_name):
  tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
  graph = pydotplus.graph_from_dot_data(tree_str)
  graph.write_png(file_name)

tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final_2.png")

### Extracting Decision Trees ###
tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)

#output
"""
|--- Glucose <= 127.50
|   |--- Age <= 28.50
|   |   |--- BMI <= 45.40
|   |   |   |--- BMI <= 30.95
|   |   |   |   |--- Pregnancies <= 7.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- Pregnancies >  7.50
|   |   |   |   |   |--- class: 1
|   |   |   |--- BMI >  30.95
|   |   |   |   |--- DiabetesPedigreeFunction <= 0.50
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- DiabetesPedigreeFunction >  0.50
|   |   |   |   |   |--- class: 0
|   |   |--- BMI >  45.40
|   |   |   |--- BloodPressure <= 99.00
|   |   |   |   |--- class: 1
|   |   |   |--- BloodPressure >  99.00
|   |   |   |   |--- class: 0
|   |--- Age >  28.50
|   |   |--- BMI <= 26.35
|   |   |   |--- BMI <= 9.65
|   |   |   |   |--- class: 1
|   |   |   |--- BMI >  9.65
|   |   |   |   |--- class: 0
|   |   |--- BMI >  26.35
|   |   |   |--- Glucose <= 99.50
|   |   |   |   |--- Glucose <= 28.50
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- Glucose >  28.50
|   |   |   |   |   |--- class: 0
|   |   |   |--- Glucose >  99.50
|   |   |   |   |--- DiabetesPedigreeFunction <= 0.56
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- DiabetesPedigreeFunction >  0.56
|   |   |   |   |   |--- class: 1
|--- Glucose >  127.50
|   |--- BMI <= 29.95
|   |   |--- Glucose <= 145.50
|   |   |   |--- Insulin <= 132.50
|   |   |   |   |--- BMI <= 28.15
|   |   |   |   |   |--- class: 0
|   |   |   |   |--- BMI >  28.15
|   |   |   |   |   |--- class: 1
|   |   |   |--- Insulin >  132.50
|   |   |   |   |--- class: 0
|   |   |--- Glucose >  145.50
|   |   |   |--- Age <= 25.50
|   |   |   |   |--- class: 0
|   |   |   |--- Age >  25.50
|   |   |   |   |--- Age <= 61.00
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- Age >  61.00
|   |   |   |   |   |--- class: 0
|   |--- BMI >  29.95
|   |   |--- Glucose <= 157.50
|   |   |   |--- Age <= 30.50
|   |   |   |   |--- BloodPressure <= 61.00
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- BloodPressure >  61.00
|   |   |   |   |   |--- class: 0
|   |   |   |--- Age >  30.50
|   |   |   |   |--- DiabetesPedigreeFunction <= 0.43
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- DiabetesPedigreeFunction >  0.43
|   |   |   |   |   |--- class: 1
|   |   |--- Glucose >  157.50
|   |   |   |--- Insulin <= 629.50
|   |   |   |   |--- DiabetesPedigreeFunction <= 0.30
|   |   |   |   |   |--- class: 1
|   |   |   |   |--- DiabetesPedigreeFunction >  0.30
|   |   |   |   |   |--- class: 1
|   |   |   |--- Insulin >  629.50
|   |   |   |   |--- class: 0
"""

###################################################
#### Extracting Python Codes of Decision Rules ####
###################################################

# works for specific scikit-learn version 
#pip install scikit-learn==0.23.1

print(skompile(cart_final.predict).to('python/code'))

#OUTPUT
"""
(((((0 if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else 0 if x[6] <=
    0.5005000084638596 else 0) if x[5] <= 45.39999961853027 else 1 if x[2] <=
    99.0 else 0) if x[7] <= 28.5 else (1 if x[5] <= 9.649999618530273 else 
    0) if x[5] <= 26.350000381469727 else (1 if x[1] <= 28.5 else 0) if x[1
    ] <= 99.5 else 0 if x[6] <= 0.5609999895095825 else 1) if x[1] <= 127.5
     else (((0 if x[5] <= 28.149999618530273 else 1) if x[4] <= 132.5 else 
    0) if x[1] <= 145.5 else 0 if x[7] <= 25.5 else 1 if x[7] <= 61.0 else 
    0) if x[5] <= 29.949999809265137 else ((1 if x[2] <= 61.0 else 0) if x[
    7] <= 30.5 else 1 if x[6] <= 0.4294999986886978 else 1) if x[1] <= 
    157.5 else (1 if x[6] <= 0.3004999905824661 else 1) if x[4] <= 629.5 else 0
    )
"""

print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

#OUTPUT
"""
SELECT CASE WHEN (x2 <= 127.5) 
THEN CASE WHEN (x8 <= 28.5) 
THEN CASE WHEN (x6 <= 45.39999961853027) 
THEN CASE WHEN (x6 <= 30.949999809265137) 
THEN CASE WHEN (x1 <= 7.5) 
THEN 0 ELSE 1 END ELSE 0 END 
ELSE CASE WHEN (x3 <= 99.0) 
THEN 1 ELSE 0 END END ELSE CASE WHEN (x6 <= 26.350000381469727) THEN CASE WHEN (x6 <= 9.649999618530273) THEN 1 ELSE 0 END ELSE CASE WHEN (x2 <= 99.5) THEN CASE WHEN (x2 <= 28.5) THEN 1 ELSE 0 END ELSE CASE WHEN (x7 <= 0.5609999895095825) THEN 0 ELSE 1 END END END END ELSE CASE WHEN (x6 <= 29.949999809265137) THEN CASE WHEN (x2 <= 145.5) THEN CASE WHEN (x5 <= 132.5) THEN CASE WHEN (x6 <= 28.149999618530273) THEN 0 ELSE 1 END ELSE 0 END ELSE CASE WHEN (x8 <= 25.5) THEN 0 ELSE CASE WHEN (x8 <= 61.0) THEN 1 ELSE 0 END END END ELSE CASE WHEN (x2 <= 157.5) THEN CASE WHEN (x8 <= 30.5) THEN CASE WHEN (x3 <= 61.0) THEN 1 ELSE 0 END ELSE 1 END ELSE CASE WHEN (x5 <= 629.5) THEN 1 ELSE 0 END END END END AS y 
FROM data

"""

print(skompile(cart_final.predict).to('excel'))

### Making Prediction With Extracted Python Codes ###

def predict_with_rules(x):
  return (((((0 if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else 0 if x[6] <=
    0.5005000084638596 else 0) if x[5] <= 45.39999961853027 else 1 if x[2] <=
    99.0 else 0) if x[7] <= 28.5 else (1 if x[5] <= 9.649999618530273 else 
    0) if x[5] <= 26.350000381469727 else (1 if x[1] <= 28.5 else 0) if x[1
    ] <= 99.5 else 0 if x[6] <= 0.5609999895095825 else 1) if x[1] <= 127.5
     else (((0 if x[5] <= 28.149999618530273 else 1) if x[4] <= 132.5 else 
    0) if x[1] <= 145.5 else 0 if x[7] <= 25.5 else 1 if x[7] <= 61.0 else 
    0) if x[5] <= 29.949999809265137 else ((1 if x[2] <= 61.0 else 0) if x[
    7] <= 30.5 else 1 if x[6] <= 0.4294999986886978 else 1) if x[1] <= 
    157.5 else (1 if x[6] <= 0.3004999905824661 else 1) if x[4] <= 629.5 else 0
    )

X.columns

x = [0, 113, 23420, 2323, 431, 5225, 23, 0]

predict_with_rules(x)
# outcome = 1 (models prediction for this sample) 

