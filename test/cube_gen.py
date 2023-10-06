from numpy import random

n = 200000

coords = random.rand(n,3)
w = 8988

fid = open('test/cube'+str(n),'w')
for idx in range(n):
    fid.write(str(idx) + ' ' + str(coords[idx,:])[1:-1] + ' ' + str(w) + '\n')
    
fid.close()
