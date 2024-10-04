#### Seaborn ####
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

df["sex"].value_counts()
sns.countplot(x=df["sex"],data=df)
plt.show()

#### For Numerical Values
sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()

df["total_bill"].head(10).plot.barh()
plt.show()


sns.scatterplot(x=df["tip"], y=df["total_bill"], hue=df["smoker"], data=df)
plt.show()