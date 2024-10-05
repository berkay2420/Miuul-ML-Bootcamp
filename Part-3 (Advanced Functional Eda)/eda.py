#### Advanced Functional Exploratory Data Analysis (EDA) ####

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["survived"].value_counts() 
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe()
df.isnull().values.any()
df.isnull().sum()

####
def check_df(dataframe,head=5):
  print("########## Shape #############")
  print(dataframe.shape)
  print("########## Types #############")
  print(dataframe.dtypes)
  print("########## Head #############")
  print(dataframe.head(head))
  print("########## Tail #############")
  print(dataframe.tail(head))
  print("########## Na #############")
  print(dataframe.isnull().sum())
  print("########## Quantities #############")
  print(dataframe.describe([0, 0.5, 0.95, 0.99, 1])).T

check_df(df)

#############################################
#### Analysis of Categorical Variables ######
#############################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

def check_categorical_variables(dataframe):
  categorical_cols = dataframe.select_dtypes(include=['category','bool','object']).columns.tolist()
  print("############ List of Caregorical Columns ############")
  print(categorical_cols)
  print("########### Na ###########")
  print(dataframe[categorical_cols].isnull().sum())
  

check_categorical_variables(df) 

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ['category','bool','object'] ]

num_but_cat = [col for col in df.columns if  df[col].nunique() < 10 and df[col].dtypes in ["int64","float64"]]
# Sayısal değerlden oluşan ama yine de kategorik olanları değişken sayısının büyüklüğüne göre belirledik

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]
# finding not optimal datas. Where column is categorical or object and have more than 20 unique classes 

categorical_cols = cat_cols + num_but_cat

categorical_cols = [col for col in categorical_cols if col not in cat_but_car]

df[categorical_cols].nunique() #number of classes each have


def cat_summary(dataframe, col_name):
  print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 
                      "Ratio": 100*dataframe[col_name].value_counts()/len(dataframe)}))
  print("############################")


cat_summary(df,"sex")

###
for col in categorical_cols:
  cat_summary(df,col)

def cat_summary(dataframe, col_name, plot=False):
  """
  Function for summarize column info, function first checks whether the given data frame
  has bool type values or not and changes bool types to integer type
  Args:
      dataframe (dataframe):
      col_name (string): 
      plot (bool, optional):  Defaults to False.
  """
  if dataframe[col_name].dtypes == "bool": ## seaborn plot doesnt accept bool values 
    dataframe[col_name] = dataframe[col_name].astype(int) #---> 1 for True and 0 for False

  print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 
                      "Ratio": 100*dataframe[col_name].value_counts()/len(dataframe)}))
  
  if plot:
    sns.countplot(x=dataframe[col_name], data=dataframe)
    plt.show(block=True)    


for col in categorical_cols:
  cat_summary(df,col, plot=True)


df["adult_male"] #---> True, False


#############################################
#### Analysis of Numerical Variables ########
#############################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

df[["age","fare"]].describe()


num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]

num_cols = [col for col in num_cols if col not in categorical_cols]

def num_summary(dataframe, numerical_col):
  quantiles = [0.05, 0.1, 0.2 ,0.3, 0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,0.99]
  print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")

for col in num_cols:
  num_summary(df,col)

###
def num_summary(dataframe, numerical_col, plot=False):
  quantiles = [0.05, 0.1, 0.2 ,0.3, 0.4 ,0.5 ,0.6 ,0.7 ,0.8 ,0.9 ,0.99]
  print(dataframe[numerical_col].describe(quantiles).T)

  if plot:
    dataframe[numerical_col].hist()
    plt.xlabel(numerical_col)
    plt.title(numerical_col)
    plt.show(block=True)

for col in num_cols:
  num_summary(df, col, plot=True)

##########################################################
#### Capturing Values and Generalizing Operations ########
##########################################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

def grab_col_names(dataframe, cat_th=10, car_th=20):
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


  cat_cols = [col for col in df.columns 
              if str(df[col].dtypes) in ['category','bool','object'] ]

  num_but_cat = [col for col in df.columns 
                 if  df[col].nunique() < 10 and df[col].dtypes in ["int64","float64"]]

  cat_but_car = [col for col in df.columns 
                 if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]

  categorical_cols = cat_cols + num_but_cat
  categorical_cols = [col for col in categorical_cols if col not in cat_but_car]
  
  num_cols = [col for col in df.columns 
              if df[col].dtypes in ["int","float"]]
  
  num_cols = [col for col in num_cols if col not in categorical_cols]

  print(f"Observations:{dataframe.shape[0]}")
  print(f"Variables:{dataframe.shape[1]}")
  print(f"categorical_cols:{len(categorical_cols)}")
  print(f"numerical_cols:{len(num_cols)}")
  print(f"categorical_but_cardinite:{len(cat_but_car)}")
  print(f"numerical_but_cardinite:{len(num_but_cat)}")

  return categorical_cols, num_cols, cat_but_car

grab_col_names(df)


#####################################
#### Target Variable Analysis #######
#####################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

for col in df.columns:
  if df[col].dtypes == "bool":
    df[col] = df[col].astype(int)

def grab_col_names(dataframe, cat_th=10, car_th=20):
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


  cat_cols = [col for col in df.columns 
              if str(df[col].dtypes) in ['category','bool','object'] ]

  num_but_cat = [col for col in df.columns 
                 if  df[col].nunique() < 10 and df[col].dtypes in ["int64","float64"]]

  cat_but_car = [col for col in df.columns 
                 if df[col].nunique() > 20 and str(df[col].dtypes) in ["category","object"]]

  categorical_cols = cat_cols + num_but_cat
  categorical_cols = [col for col in categorical_cols if col not in cat_but_car]
  
  num_cols = [col for col in df.columns 
              if df[col].dtypes in ["int","float"]]
  
  num_cols = [col for col in num_cols if col not in categorical_cols]

  print(f"Observations:{dataframe.shape[0]}")
  print(f"Variables:{dataframe.shape[1]}")
  print(f"categorical_cols:{len(categorical_cols)}")
  print(f"numerical_cols:{len(num_cols)}")
  print(f"categorical_but_cardinite:{len(cat_but_car)}")
  print(f"numerical_but_cardinite:{len(num_but_cat)}")

  return categorical_cols, num_cols, cat_but_car


categorical_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

###
df.groupby("sex")["survived"].mean() #sex variables with their survival rate

def target_summary_with_cat(dataframe, target, categorical_col):
  print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df, "survived", "sex")

target_summary_with_cat(df, "survived", "pclass") # gives survival rate

for col in categorical_cols:
  target_summary_with_cat(df, "survived", col)

#########################################################
#### Target Variable Analysis for Numrical Values #######
#########################################################
df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
  print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(target)[numerical_col].mean(),}))

##Other way
def target_summary_with_num(dataframe, target, numerical_col):
  print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")



for col in num_cols:
  target_summary_with_num(df, "survived", col)

##################################
#### Correlation Analysis ########
##################################