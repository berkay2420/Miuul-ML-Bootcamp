def makeList(message):
  message_list = []

  for char in message:
    message_list.append(char)

  for index in range(len(message_list)):
    if (index % 2) == 0:
      message_list[index] = message_list[index].upper()

  new_sentence = ''.join(message_list)

  
  return new_sentence


message = input("write your message:\n")

print(makeList(message))