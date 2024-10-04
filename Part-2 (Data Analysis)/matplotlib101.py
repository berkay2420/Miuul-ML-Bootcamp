import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()

#### plot ####
x = np.array([1,8])
y = np.array([0,150])

plt.plot(x,y)
plt.show()

plt.plot(x,y, 'o')
plt.show()

x = np.array([2,4,6,8,10])
y = np.array([1,3,5,7,9])

plt.plot(x,y)
plt.plot(x,y, 'o')
plt.show()


#### marker ####
y = np.array([13,23,11,90])
plt.plot(y, marker='o')
plt.show()

plt.plot(y, marker='x')
plt.show()

#### line ####
y = np.array([13,23,11,90])
plt.plot(y, linestyle="dashed", color="r")
plt.show()

x = np.array([2,45,6,1,10])
y = np.array([1,3,32,7,9])
plt.plot(x, label="x verisi")
plt.plot(y, label="y verisi")
plt.xlabel("X Ekseni")
plt.ylabel("Y Ekseni")
plt.title("Ana başlık")
plt.grid()
plt.legend(loc='upper center')
plt.show()


#### Subpots ####

##plot 1
x = np.array([2,45,6,1,10,13,21,313,132,32,323,2])
y = np.array([2,45,6,1,4,3,212,33,12,32,33,21])
plt.subplot(1,2,1) #subplot(nrows, ncols, index, **kwargs)
plt.title("1")
plt.plot(x,y)

## plot 2
x = np.array([2,45,6,1,10,13,21,313,132,32,323,2])
y = np.array([2,45,6,1,4,3,212,33,12,32,33,21])
plt.subplot(1,2,2) #subplot(nrows, ncols, index, **kwargs)
plt.title("1")
plt.plot(x,y)
plt.show()