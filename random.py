from random import randint
from linereader import dopen

length = 1045
file_path = 'input/fold0/Alldata.txt'

file_path = dopen(filename)
random_line = file.getline(randint(1, length))