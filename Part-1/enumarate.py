students = ["John","Cartman","Bob","Vanessa"]

# enumerate = string'in kendisi ve index numaraları için 

# for index, student in enumerate(students):
#   print(index,student)

def divide_students(student_list):
  groups = [[],[]]
  for index, student in enumerate(student_list):
    if index % 2 == 0:
      groups[0].append(student)
    else:
      groups[1].append(student)
  
  return groups

print(divide_students(students))

