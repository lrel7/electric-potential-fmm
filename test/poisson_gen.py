import numpy as np
fid = open('test/point_charge', 'w')
idx = 0
d = 0.05

def writefile(x, y, z, w):
    global idx
    fid.write(str(idx) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(w) + '\n')
    idx = idx + 1

for x in np.arange(0, 1, d):
    for y in np.arange(0, 1, d):
        for z in np.arange(0, 1, d):
            if x == 0.8 and y == 0.8 and z == 0.8:
                w = 1
            else:
                w = 0
            writefile(x, y, z, w)