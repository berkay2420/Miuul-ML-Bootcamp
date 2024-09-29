#LAMBDA
sum = lambda a,b: a + b
print(sum(1,2))

#MAP
salaries = [1000, 2000, 3000, 4000]

def new_salary(x):
  return x*20 / 100 + x
for salary in salaries:
  print(new_salary(salary))

new_salary_list = list(map(new_salary, salaries))  #takes function and a list as an arg
print(new_salary_list)

lambda_map_salary_list = list(map(lambda x: x * 20 / 100 + x, salaries))
print(lambda_map_salary_list)


#FILTER
list_score = [1,2,3,4,32,5,23,21]
list(filter(lambda x: x % 2 == 0, list_score)) 

#Reduce

from functools import reduce

list_score= [1,23,4,5]
reduce(lambda a,b: a + b, list_score)
