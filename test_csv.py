import csv
import sys
import random

fname = sys.argv[1] + '.csv'

f = open(fname, 'w', newline='\n')
writer = csv.writer(f)

for i in range(10):
    writer.writerow([random.randint(0, 10), 10])
    

