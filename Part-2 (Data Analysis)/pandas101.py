#### Pandas Series ####
from re import A
from traceback import print_tb
import pandas as pd
from regex import D

s = pd.Series([12,3,221,31,4])
type(s)
s.index
s.dtype

#### Reading Data ####

import pandas as pd

# advertising.csv dosyasını oku
df = pd.read_csv('C:/Users/User/Desktop/Miuul-ML/Part-2 (Data Analysis)/advertising.csv')
print(df.head())

##############
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
print(df.head())
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

df["sex"].value_counts() # ---> male=577  female=314


#### Selection in Pandas ####
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")
df.head()

# df.drop(delete_indexes, axis=0, inplace=True) "inplace=True" for making change permanent

############
df["age"].head()
df.age.head()

df.index = df["age"]  # sort data by age 
df.index

df.drop("age", axis=1).head()
df.drop("age", axis=1, inplace=True)

#########
df.index

df["age"] = df.index  # adding age to the variables
df.head(10)

######
df.drop("age", axis=1, inplace=True)
df.reset_index().head() #deletes the index and adds it to the variables

###########
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

"age" in df # True

df["age"].head() # This is a pandas series
type(df["age"].head())

df[["age"]].head() # This is a dataframe
type(df[["age"]].head()) 

df[["age","alive"]].head()

col_names = ["age","embarked","alive"]
df[col_names].head()

df["age2"] = df["age"]**2  # Adding New column
df.head()

df.loc[:, df.columns.str.contains("age")].head() # colums that contain the substring "age"

df.loc[:, ~df.columns.str.contains("age")].head() #used "~" colums that doen't contain the substring "age"

#### loc & iloc ####
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

# iloc: integer based selection
df.iloc[0:3] #first three row (0-1-2)
df.iloc[0,0] #finding items 

# loc: label based selection
df.loc[0:3] #first four row (0-1-2-3)

df.iloc[0:3, "age"] #error can't read "age" label
df.loc[0:3, "age"] #succes first 4 rows of the age label

col_names = ["age","embarked","alive"]
df.loc[0:4, col_names]  

#### Conditional Selection ####
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"]>50].head(5)
df[df["age"]>50]["age"].count() # older than 50 


df.loc[df["age"]>50, ["class","age"]].head() # classes of  older than 50 

df.loc[(df["age"]>50) & (df["sex"]=="male"), ["class","age","sex"]].head() # males over 50 years old

col_list=["class","age","sex","embark_town"]
df.loc[(df["age"]>50) 
       & (df["sex"]=="male") 
       & ((df["embark_town"]=="Cherbourg") | (df["embark_town"]=="Southampton")),
       col_list].head()

df["embark_town"].value_counts()

### Aggretion and Grouping
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.loc[:,["sex","age"]]

df["age"].mean()
df.groupby("sex")["age"].mean()  # Group by sex female=27.915709 male=30.726645

df.groupby("sex").agg({"age":"mean"}) # More popular usage

df.groupby("sex").agg({"age":["mean","sum",]})

df.groupby("sex").agg({"embark_town"})

###
df.groupby("sex").agg({"age":["mean","sum"],"embark_town":["count"]})

###
df.groupby("sex").agg({"age":["mean","sum"],"survived":["mean"]}) 

###
df.groupby(["sex","embark_town"]).agg({"age":["mean","sum"],"survived":["mean"]})
##Cinsiyetlierine göre hangi farklı limanlardan katılanların yaşlarının toplamı ortalaması
# ve hayatta kalma olasılıkları 

###
df.groupby(["sex","class","embark_town",]).agg({"age":["mean"],"survived":["mean"]})

###
df.groupby(["sex","embark_town","class",]).agg({
  "age":["mean"],
  "survived":["mean"],
  "sex":["count"]})

#### Pivot Table ####
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived","sex","embarked")  #pivot_table(values, index, columns, agg_func(default->mean))


df.pivot_table("survived","sex",["embarked","class"]).head(3)

###
df["new_age"] = pd.cut(df["age"], [0,10,18,25,40,90]) #convert numarical data to categorical data also qcut
df.head() # now we have new_age column 

age_labels=["child","teen","young adult","adult","old"]
df["new_age"] = pd.cut(df["age"], bins=[0,10,18,25,40,90], labels=age_labels) 
df.head(5)

df.pivot_table("survived","sex",["new_age","class"]).head(3)

### display settings

pd.set_option('display.width',500)

#### Apply & Lambda ####
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)
df = sns.load_dataset("titanic")
df.head(10)

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 3


for col in df.columns:
  if "age" in col:
    df[col] = df[col] /10

df.head()

### apply
df[["age","age2","age3"]].apply(lambda x:x**2).head()  # applies to given colums

df.loc[:,df.columns.str.contains("age")].apply(lambda x:x**2).head()

df.loc[:,df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()

def standart_scaler(col_name):
  return (col_name - col_name.mean()) / col_name.std()

df.loc[:,df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:,df.columns.str.contains("age")]=df.loc[:,df.columns.str.contains("age")].apply(standart_scaler)

df.head(5)

#### Join ####
import numpy as np
import pandas as pd
m = np.random.randint(1,30,size=(5,3))
df1=pd.DataFrame(m,columns=["var1","var2","var3"])

df2 = df1 + 99 #applies to every cell
df1.head()
df2.head()

pd.concat([df1,df2])

pd.concat([df1,df2], ignore_index=True) # new dataframe with 10 indexes

pd.concat([df1,df2], ignore_index=True, axis=1) # new dataframe with 6 columns

### Merge
df1 = pd.DataFrame({'employees': ["john","marry","bob"],
                   'group': ["accounting","it","HR"]})

df2 = pd.DataFrame({
  'employees': ["marry","john","bob"],
  'start_date': [2010,2012,2015]
  })
pd.merge(df1,df2)
pd.merge(df1,df2, on="employees")

