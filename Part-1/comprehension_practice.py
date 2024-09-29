#Veri setineki değişkenlerin isimlerini değiştirme

#before
#['total','speeding','alcohol','previous','ins_premium','ins_loses','abbrev']

#after
#['TOTAL','SPEDING','ALCOHOL', 'PREVIOUS', 'INS_PREMIUM','INS_LOSES','ABBREV']

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

df.columns = [col.upper() for col in df.columns]
df.columns

##Adding 'FLAG' to words starts with "INS"

["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns  ]

##################################################################################
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

numerical_cols = [col for col in df.columns if df[col].dtype != "O"] # "o" stands for object


func_list=['mean','min','max','var']

new_dict = {col_name:func_list for col_name in numerical_cols }
df[numerical_cols].head()


df[numerical_cols].agg(new_dict) #agg() for getting results from func_list applied for numerical columns


########################################

kume1= set(["data","python"])
kume2=set(["data","function","qcut","lambda","python","miuul"])

def difference(set1,set2):
  difference_lst=[  ]
  for item in set2:
    if item  not in set1:
      difference_lst.append(item)
  return difference_lst

print(difference(kume1,kume2))

###########################################################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

og_list=["abbrev","no_previous"]

new_cols = [ col for col in df.columns if col not in og_list ]
print(new_cols)

new_df=df[new_cols]

print(new_df.head())