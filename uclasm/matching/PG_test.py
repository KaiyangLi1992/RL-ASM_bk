import sys

my_dict = {'key': 'value'}
my_tuple = ('key', 'value')
my_list  = ['key', 'value']

print(sys.getsizeof(my_dict))  # 打印字典的大小
print(sys.getsizeof(my_tuple)) # 打印元组的大小
print(sys.getsizeof(my_list)) # 打印元组的大小