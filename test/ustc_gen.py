import numpy as np
fid = open('test/ustc', 'w')
idx = 0

w = 0.01
d = 0.006

def writefile(x, y):
    global idx
    fid.write(str(idx) + ' ' + str(x) + ' ' + str(y) + ' ' + str(0) + ' ' + str(w) + '\n')
    idx = idx + 1

# U
for y in np.arange(0.7, 0.4, -d):
    for x in np.arange(0, 0.08, d):
        writefile(x, y)
    for x in np.arange(0.17, 0.24, d):
        writefile(x, y)
for y in np.arange(0.4, 0.2, -d):
    for x in np.arange(0, 0.24, d):
        writefile(x, y)

# S
for y in np.arange(0.7, 0.6, -d):
    for x in np.arange(0.26, 0.49, d):
        writefile(x, y)
for y in np.arange(0.6, 0.5, -d):
    for x in np.arange(0.26, 0.3225, d):
        writefile(x, y)
for y in np.arange(0.5, 0.4, -d):
    for x in np.arange(0.26, 0.49, d):
        writefile(x, y)
for y in np.arange(0.4, 0.3, -d):
    for x in np.arange(0.4275, 0.49, d):
        writefile(x, y)
for y in np.arange(0.3, 0.2, -d):
    for x in np.arange(0.26, 0.49, d):
        writefile(x, y)

# T
for y in np.arange(0.7, 0.5, -d):
    for x in np.arange(0.51, 0.74, d):
        writefile(x, y)
for y in np.arange(0.5, 0.2, -d):
    for x in np.arange(0.61, 0.64, d):
        writefile(x, y)

# C
for y in np.arange(0.7, 0.6, -d):
    for x in np.arange(0.76, 1, d):
        writefile(x, y)
for y in np.arange(0.6, 0.3, -d):
    for x in np.arange(0.76, 0.84, d):
        writefile(x, y)
for y in np.arange(0.3, 0.2, -d):
    for x in np.arange(0.76, 1, d):
        writefile(x, y)

