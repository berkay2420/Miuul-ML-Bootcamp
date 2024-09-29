dict = {
  'a':1,
  'b':2,
  'c':3,
  'd':4
}

dict.keys()
dict.values()
dict.items()

{key: value ** 2 for (key,value) in dict.items()}

numbers = range(10)

new_dict = {}

for number in numbers:
  if number % 2 == 0:
    new_dict[number]= number ** 2

print(new_dict)

{n: n ** 2 for n in numbers if n % 2 == 0}