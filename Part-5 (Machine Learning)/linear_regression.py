####################################################
#### Sales Prediction With Linear Regression #######
####################################################
import bottleneck
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x:'%2.f' % x)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict

df = pd.read_csv("advertising.csv")
df.head()
df.shape

X = df[["TV"]]
y = df[["sales"]]

#### Simple Linear Regression Model ####

reg_model = LinearRegression().fit(X,y)

# y_hat = b + wx

# bias (b - sabit)
reg_model.intercept_[0]

# TV'nin katsayısı (w1)
reg_model.coef_[0][0]


#### Prediction ####

reg_model.intercept_[0] + reg_model.coef_[0][0]*150


graph = sns.regplot(x=X,y=y, scatter_kws={'color':'b','s':9},
                    ci=False, color="r")
graph.set_title(f"Model Equation: Sales = {round(reg_model.intercept_[0],2)} + TV*{round(reg_model.coef_[0][0], 2)}")
graph.set_ylabel("Satis Sayisi")
graph.set_xlabel("TV Harcamalari")
plt.xlim(-10,310)
plt.ylim(bottom=0)
plt.show()

######
#MSE
y_pred = reg_model.predict(X) # model trying to gueess depended value from given independent values

mean_squared_error(y, y_pred)
#10.512652915656757
y.mean()
y.std()

#RMSE
np.sqrt(mean_squared_error(y, y_pred))
# 3.2423221486546887

#MAE
mean_absolute_error(y, y_pred)
# 3.2423221486546887

#R-SQUARE
reg_model.score(X,y)
# 0.611875050850071
# Bağımsız değişkenin bağımlı değişkendeki değişikliği açıklayabilme yüzdesi.

############################################
##### Multiple Linear Regression Model #####
############################################

df = pd.read_csv('advertising.csv')

X = df.drop("sales", axis=1)
y = df[["sales"]]

## Model

# split with holdout
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.20,
                                                    random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)


bias = reg_model.intercept_
# 2.90794702

weights = reg_model.coef_[0]
# [0.0468431 , 0.17854434, 0.00258619]

#Prediction 
# TV:30
# newspaper:40
# radio:10

# b + wx

# multiple linear regression
def make_predictions(values=[]):

  for i in range(len(weights)) :
    prediction = bias + weights[i] * values[i]
    prediction += prediction
  print(f"Precitions for sales {prediction}")
  
make_predictions(values=[30,40,10])

##
new_data = [[30],[10],[40]]
new_data = pd.DataFrame(new_data).T

y_pred = reg_model.predict(new_data)

### Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

### Train R-Squared
reg_model.score(X_train,y_train)

### Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

### Test R-Squared
reg_model.score(X_test,y_test)


# 10 fold Cross Validation RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv=10,
                                 scoring="neg_mean_squared_error")))