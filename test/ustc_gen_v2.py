import numpy as np
fid = open('test/ustc', 'w')
idx = 0

d = 0.006

def writefile(x, y):
    global idx
    fid.write(str(idx) + ' ' + str(x) + ' ' + str(y) + ' ' + str(0) + ' ' + str(w) + '\n')
    idx = idx + 1

# U
w = 4103904
for y in np.arange(0.7, 0.4, -d):
    for x in np.arange(0, 0.06, d):
        writefile(x, y)
    for x in np.arange(0.15, 0.21, d):
        writefile(x, y)
for y in np.arange(0.4, 0.2, -d):
    for x in np.arange(0, 0.21, d):
        writefile(x, y)
print(idx)
print("\n")

# S
w = 5083456
for y in np.arange(0.7, 0.6, -d):
    for x in np.arange(0.28, 0.46, d):
        writefile(x, y)
for y in np.arange(0.6, 0.5, -d):
    for x in np.arange(0.28, 0.3225, d):
        writefile(x, y)
for y in np.arange(0.5, 0.4, -d):
    for x in np.arange(0.28, 0.46, d):
        writefile(x, y)
for y in np.arange(0.4, 0.3, -d):
    for x in np.arange(0.4275, 0.46, d):
        writefile(x, y)
for y in np.arange(0.3, 0.2, -d):
    for x in np.arange(0.28, 0.46, d):
        writefile(x, y)

print(idx)
print("\n")

# T
w = 7366845
for y in np.arange(0.7, 0.5, -d):
    for x in np.arange(0.54, 0.72, d):
        writefile(x, y)
for y in np.arange(0.5, 0.2, -d):
    for x in np.arange(0.61, 0.63, d):
        writefile(x, y)
        
print(idx)
print("\n")

# C
w = 4837218
for y in np.arange(0.7, 0.6, -d):
    for x in np.arange(0.78, 1, d):
        writefile(x, y)
for y in np.arange(0.6, 0.3, -d):
    for x in np.arange(0.78, 0.85, d):
        writefile(x, y)
for y in np.arange(0.3, 0.2, -d):
    for x in np.arange(0.78, 1, d):
        writefile(x, y)
        
print(idx)
print("\n")

