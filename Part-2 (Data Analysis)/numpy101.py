list1= [1,2,3,4]
list2=[2,3,4,5]
list3=[]

## basic pyton
for i in range(0,len(list1)):
  list3.append(list1[i]*list2[i])
print(list3)

## numpy
import numpy as np
list1=np.array([1,2,3,4,8])
list2=np.array([2,3,4,5,6])
list3=list1* list2
print(list3)

# numpy array is faster than list because np array stores homogenous data on the other hand
# list stores heterogeneous data (a python list can store vlaues with different type at the same time)


##Creating Arrays In Numpy

import numpy as np
arr = np.array([1, 2, 4])
print(type(arr))

zero_array = np.zeros(10, dtype=int)
random_array = np.random.randint(0, 10, size=5)
special_array = np.random.normal(10, 5, (2,3)) # mean(ortalama), standart deviation(standart sapma), shape

## Attributes of Numpy Arrays
import numpy as np
array1= np.random.randint(10, size=5)

array1.ndim 
array1.shape
array1.size
array1.dtype

## Reshaping

array = np.random.randint(1,10, size=9)
array.reshape(3,3)


## Index Selection

import numpy as np
arr = np.random.randint(10, size=10)
arr[9]
arr[0:5]

arr[9] = 10
arr[9]

arr2 = np.random.randint(10, size=(3,5))
arr2
arr2[0,0]
arr2[0,0] = 22.9
arr2[:, 0]
arr2[1, :]

arr2
arr2[0:3,3:5]   # [includes, not includes]


## Fancy Index
import numpy as np

arr = np.arange(0,36,3)
arr

catch = [1,2,3]
arr[catch]

#### Conditions on Numpy ####
import numpy as np

arr = np.array([1,2,3,4,5])
arr < 3 # ----> array([ True,  True, False, False, False])
arr[arr<3] # -----> array([1, 2])


#### Mathematical Operations ####
import numpy as np
arr = np.array([1,2,3,4,5])

arr / 5  # divides every item in array
arr ** 3
arr + arr ** 2 

np.subtract(arr, 2)
np.add(arr,5)
np.mean(arr)
np.sum(arr)
np.min(arr)
np.max(arr)
np.var(arr)

## More advanced math in numpy

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5,1],[1,3]])
b = np.array([12,10])

np.linalg.solve(a,b)

  