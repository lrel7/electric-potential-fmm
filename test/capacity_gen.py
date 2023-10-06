import numpy as np

def writefile(x, y, z):
    global idx
    fid.write(str(idx) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + ' ' + str(w) + '\n')
    idx = idx + 1

# plane
idx = 0
w = 1797510

fid = open('test/plane', 'w')
z = 0.2
d = 0.02
for x in np.arange(0, 1, d):
    for y in np.arange(0, 1, d):
        writefile(x, y, z)
w = -1797510
z = 0.8
for x in np.arange(0, 1, d):
    for y in np.arange(0, 1, d):
        writefile(x, y, z)
fid.close()
