from enum import unique
from re import A
import pandas as pd
import seaborn as sns
df=sns.load_dataset("titanic")
df.head()

df["sex"].value_counts()

def get_unique_count(col):
  unique_list= []
  return unique_list.count()

for col in df.columns:
  print(f"{col} column has {get_unique_count(col)} unique values")

# unique values by column
df.nunique()

df["pclass"].unique()
df[["pclass","parch"]].nunique()

# type check and change
df.dtypes["embarked"]
df["embarked"].astype('category')
df.dtypes["embarked"]

df[df["embarked"]=="C"]

df[df["embarked"] != "S"]

df[ (df["age"] < 30) & (df["sex"] == "female") ]


df[(df["fare"]>500) | (df["age"] > 70)].head(5)


# finding null values
df.isnull().sum()

#checking if there is null values in data
df.isnull().values.any()

#dropping column
df.drop("who", axis=1)

# filling null values
mod_value = df["deck"].mode()[0]
df.loc[df["deck"].isnull(), "deck"] = mod_value

df["deck"].fillna(df["deck"].mode()[0], inplace=True)

df["age"].fillna(df["age"].median())

###
df.pivot_table("survived","pclass","sex",["sum","count","mean"])

df.groupby(['pclass', 'sex'])['survived'].agg(['sum', 'count', 'mean'])

###
df["age_flag"]= pd.cut(df["age"])

###
def label_age(x):
  if x < 30:
    return 1
  else:
    return 0
  
df["age_flag"] = df.loc[:,"age"].apply(label_age)

df.head(5)

df["age_flag"] = df.loc[:,"age"].apply(lambda x:  1 if x <30 else 0)

## tips data frame
import pandas as pd
import seaborn as sns

df = sns.load_dataset("tips")
df.head(5)

# question 18
df.groupby("time").agg({"total_bill":["sum","min","max","mean"]})

#q19
df.groupby(["time","day"]).agg({"total_bill":["sum","min","max","mean"]})

#q20


#q21
df.loc[(df["size"] < 3) & (df["total_bill"] > 10), ["size","total_bill"]].mean().head()

#q22
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

df[["total_bill_tip_sum","total_bill","tip"]]

#q22
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
new_df = df["total_bill_tip_sum"].nlargest((30))
print(new_df)

new_df = df.nlargest(30, "total_bill_tip_sum")

print(new_df)