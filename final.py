import sys
import numpy
import time
from helper import *

assert (len(sys.argv) == 4), "The format should be \n [script.py] [filename] [n_crit] [theta]"

filename = sys.argv[1]
particles = read_particle(filename)

capacity = int(sys.argv[2])
theta = float(sys.argv[3])

# build tree
tic = time.time()
root = Cell(capacity)
root.x, root.y, root.z = 0.5, 0.5, 0.5
root.r = 0.5
cells = build_tree(particles, root, capacity)
toc = time.time()
t_src = toc - tic

# P2M: particle to multipole
tic = time.time()
leaves = []
get_leaf_cell_multipole(particles, 0, cells, leaves, capacity)
toc = time.time()
t_P2M = toc - tic

# M2M
tic = time.time()
upward_sweep(cells)
toc = time.time()
t_M2M = toc - tic

# evaluate the potentials
tic = time.time()
eval_potential(particles, cells, capacity, theta)
toc = time.time()
t_eval = toc -tic

# print info
print(filename + '-serial' + '-non-vectorized-treecode')
print(len(filename + '-serial' + '-non-vectorized-treecode')*'-')
print("     N = %i" % len(particles))
print("n_crit = %i" % capacity)
print(" theta = %.2f" % theta)

# calculate the error
u_direct = numpy.loadtxt('test/'+filename+'_result')
u_tree = numpy.asarray([particle.u for particle in particles])
l2_err(u_direct, u_tree)

# print benchmark
t_tree = t_src + t_P2M + t_M2M + t_eval
print(28*'-')
print("time elapsed: %f s" % t_tree)
print("build tree: %f, %.2f %%" % (t_src, t_src/t_tree))
print("P2M       : %f, %.2f %%" % (t_P2M, t_P2M/t_tree))
print("M2M       : %f, %.2f %%" % (t_M2M, t_M2M/t_tree))
print("eval u  : %f. %.2f %%" % (t_eval, t_eval/t_tree))

plot_particles(particles)
plot_err(u_direct, u_tree)