salaries = [1000,2000,3000,4000]

new_salaries = [salary * 2 if salary < 3000 else salary * 3 for salary in salaries]

[salary * 2 for salary in salaries]

new_salaries_3 = [salary *2 for salary in salaries if salary < 2500]


print(new_salaries_3)

students = ["john", "Bob","Adam","Tim"]

unwanted_students = ["John", "Bob"]

[name.lower() if name in unwanted_students else name.upper() for name in students]
