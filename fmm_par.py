import sys
import numpy
import time
from utility_par import *

assert (len(sys.argv) == 4), "The format should be \n [script.py] [filename] [n_crit] [theta]"

filename = sys.argv[1]
particles = read_particle(filename)

capacity = int(sys.argv[2])
theta = float(sys.argv[3])

# direct sum
t_start = time.time()
# u_direct = direct_sum(particles)
t_direct = time.time() - t_start
print("T(direct_sum) = %fs" % (t_direct))

# fmm
t_start = time.time()

# build tree
root = Cell(capacity)
root.x, root.y, root.z = 0.5, 0.5, 0.5
root.r = 0.5
cells = build_tree(particles, root, capacity)

# P2M: particle to multipole
leaves = []
get_leaf_cell_multipole(particles, 0, cells, leaves, capacity)

# M2M
upward_sweep(cells)

# evaluate the potentials
eval_potential(particles, cells, capacity, theta)

t_fmm = time.time() - t_start

# print info
print(filename + '-serial' + '-non-vectorized-treecode')
print(len(filename + '-serial' + '-non-vectorized-treecode')*'-')
print("     N = %i" % len(particles))
print(" sigma = %i" % capacity)
print(" theta = %.2f" % theta)
print("T(fmm) = %fs" % t_fmm)

# calculate the error
# u_tree = numpy.asarray([particle.u for particle in particles])
# l2_err(u_direct, u_tree)

plot_particles(particles)
# plot_potential(particles)
# plot_err(u_direct, u_tree)