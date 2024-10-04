## If the variable in data is categorical ("Yes/No," "Male/Female," or "Types of Products") 
# then we must use bar graph ["countplat"(seaborn), "barplot"(matplotlib)]

## If the variable in data is numerical then we must use "hist" or "boxplot"

#### Categorical Data Visualization ####
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind='bar')
plt.show()

#### Numerical Data Visualization ####
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

plt.hist(df["age"])
plt.show()

plt.boxplot(df["fare"])
plt.show()

