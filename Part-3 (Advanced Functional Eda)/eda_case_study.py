import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/persona.csv")
df.head()
df.tail()
df.describe()
df.value_counts()
df.columns
df.index
df.isnull().values.any()
df.isnull().sum()
df.info()

###1-2
df["SOURCE"].nunique()
df['SOURCE'].unique()
df["SOURCE"].value_counts()

###3-4
df["PRICE"].nunique()
df["PRICE"].value_counts()
df["COUNTRY"].value_counts()

###5
df.groupby("COUNTRY")["PRICE"].count()

###6
df.groupby("COUNTRY")["PRICE"].sum()

###7
df.groupby("SOURCE")["PRICE"].count()

###8-9
df.groupby("COUNTRY")["PRICE"].mean()
df.groupby("SOURCE")["PRICE"].mean()

####10
df.groupby(["COUNTRY","SOURCE"])["PRICE"].mean()

###Task-2 (COUNTRY, SOURCE, SEX, AGE kırılımındaortalama kazançlar nedir?)
df.groupby(["COUNTRY","SOURCE","SEX","AGE"])["PRICE"].mean().head()

##Task-3 (ÇıktıyıPRICE’agöre sıralayınız)
agg_df = df.groupby(["COUNTRY","SOURCE","SEX","AGE"])["PRICE"].mean().sort_values(ascending=False)

###Task-4 (Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler indexisimleridir. Bu isimleri değişken isimlerine çeviriniz)
agg_df = agg_df.reset_index()

###Task-5 (Age değişkenini kategorik değişkene çeviriniz ve agg_df’ekleyiniz)
age_labels = ["0_18", "19_23", "24_30", "31_40", "41_70"]
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], labels=age_labels, bins=[0, 18, 23, 30, 40, 70])
agg_df.head()

###Task-6 (Yeni seviye tabanlı müşterileri (persona) tanımlayınız.)

def create_persona(dataframe, ):
  customer_level_based = []

  for i in range(len(dataframe)):
    country = str(dataframe.loc[i, "COUNTRY"]).upper()
    source = str(dataframe.loc[i, "SOURCE"]).upper()
    sex = str(dataframe.loc[i,"SEX"]).upper()
    age_cat = str(dataframe.loc[i,"AGE_CAT"]).upper()
    
    customer_level_based.append(f"{country}_{source}_{sex}_{age_cat}")

  dataframe["customer_level_based"] = customer_level_based

  return dataframe["customer_level_based"]

##solution with list comprehension
cols = ["COUNTRY","SOURCE","SEX","AGE_CAT"]
agg_df["customer_level_based"] = agg_df[cols].apply(lambda row: '_'.join(row.astype(str).str.upper()), axis=1)


agg_df["customer_level_based"] = create_persona(agg_df)
     
agg_df[["customer_level_based","PRICE"]]

###Task-7 (Yeni müşterileri (personaları) segmentlere ayırınız.)
agg_df["SEGMENT"] = pd.cut(agg_df["PRICE"], labels=["D","C","B","A"], bins=[0, 25, 35, 45, 65])

agg_df[["customer_level_based","PRICE","SEGMENT"]]

agg_df.groupby("SEGMENT").agg({"PRICE":["mean","sum","max"]}).head()

###Task-8 (Yeni gelen müşterileri sınıflandırıp, ne kadar gelir getirebileceklerini  tahmin ediniz.)
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customer_level_based"] == new_user]