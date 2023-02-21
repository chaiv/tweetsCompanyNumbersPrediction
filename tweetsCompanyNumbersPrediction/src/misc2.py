'''
Created on 15.02.2023

@author: vital
'''


list1 = [1, 2, 3]
list2 = [('a', 5), ('b', 7, 9), ('c', 8, 10, 11)]

result = [(x,) + y for x, y in zip(list1, list2)]

print(result)
