import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from numba import jit

class Point(): 
    def __init__(self, coord = [], domain = 1.0) -> None:
        if coord: 
            assert len(coord) == 3, "The program works in 3d space."
            self.x = coord[0]
            self.y = coord[1]
            self.z = coord[2]
        # randomly generate coord
        else: 
            self.x = domain * np.random.random()
            self.y = domain * np.random.random() 
            self.z = domain * np.random.random()
    
    # calculate the dist between 2 points
    def distance(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

class Particle(Point): 
    # w: weight of the particle
    # u: potential
    def __init__(self, coord=[], domain=1, w = 1.0) -> None:
        Point.__init__(self, coord, domain)
        self.w = w
        self.u = 0

class Cell():
    def __init__(self, sigma) -> None:
        self.nleaf = 0 # number of particles in the cell
        self.leaf = np.zeros(sigma, dtype = np.int32) # array of particles in the cell
        self.clist = 0
        self.child = np.zeros(8, dtype = np.int32) # array of children
        self.parent = 0
        self.x = self.y = self.z = 0 # coord of the center of the cell
        self.r = 0 # radius of the cell
        self.multipole = np.zeros(10, dtype = np.float32)

    def distance(self, other):
        return np.sqrt((self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2)


# given targets list, return the array of each target's distance from the point
def distance(targets_coord, point):
    return np.sqrt((targets_coord[0] - point.x)**2 + (targets_coord[1] - point.y)**2 + (targets_coord[2] - point.z)**2)

def add_child(octant, p, cells, sigma):
    cells.append(Cell(sigma)) # create a new cell and append it to the cells list
    c = len(cells) - 1 # child n.o. of the new cell
    # geometric relationship between parent and child
    cells[c].r = cells[p].r / 2
    cells[c].x = cells[p].x + cells[c].r * ((octant & 1) * 2 - 1)
    cells[c].y = cells[p].y + cells[c].r * ((octant & 2) - 1)
    cells[c].z = cells[p].z + cells[c].r * ((octant & 4) / 2 - 1)
    # establish mutual reference in the cells list
    cells[c].parent = p
    cells[p].child[octant] = c
    cells[p].clist = (cells[p].clist | (1 << octant))
    # print('+++cell {} is created as a child of cell {}'.format(c, p))

def split_cell(particles, p, cells, sigma):
    # print('======start the split of cell {}======'.format(p))
    # loop over the particles in the parent cell that you want to split
    for l in cells[p].leaf:
        octant = (particles[l].x > cells[p].x) + ((particles[l].y > cells[p].y) << 1) \
               + ((particles[l].z > cells[p].z) << 2)   # finding the particle's octant
        # if there is not a child cell in the particle's octant, then create one
        if not cells[p].clist & (1 << octant):
            add_child(octant, p, cells, sigma)
        # reallocate the particle into the child cell
        c = cells[p].child[octant] 
        cells[c].leaf[cells[c].nleaf] = l # append the particle into the leaf list
        cells[c].nleaf += 1 # increment the number of the leaf
        # print('>>>particle {} is reallocated in cell {}'.format(l, c))
        # check if the child reach sigma
        if cells[c].nleaf >= sigma:
            split_cell(particles, c, cells, sigma)
    # print('======end split cell {}======'.format(p))
 
def build_tree(particles, root, sigma):
    # set root cell
    cells = [root]       # initialize the cells list

    # build tree
    n = len(particles)
    for i in range(n):
        # traverse from the root down to a leaf cell
        curr = 0
        while cells[curr].nleaf >= sigma:
            cells[curr].nleaf += 1
            octant = (particles[i].x > cells[curr].x) + ((particles[i].y > cells[curr].y) << 1) \
                   + ((particles[i].z > cells[curr].z) << 2)
            # if there is no child cell in the particles octant, then create one
            if not cells[curr].clist & (1 << octant):
                add_child(octant, curr, cells, sigma)
            curr = cells[curr].child[octant]
        # allocate the particle in the leaf cell
        cells[curr].leaf[cells[curr].nleaf] = i # append the particle into the current cell
        cells[curr].nleaf += 1 # increment the number of particles in the current cell
        # print('particle {} is stored in cell {}'.format(i, curr))
        # check whether to split or not
        if cells[curr].nleaf >= sigma:
            split_cell(particles, curr, cells, sigma)
    
    return cells

def get_leaf_cell_multipole(particles, p, cells, leaves, sigma):
    # if the current cell p is not a leaf cell, then recursively traverse down
    if cells[p].nleaf >= sigma:
        for c in range(8):
            if cells[p].clist & (1 << c): # if the child exists
                get_leaf_cell_multipole(particles, cells[p].child[c], cells, leaves, sigma)
    # otherwise cell p is a leaf cell
    else:
        # loop in leaf particles
        for i in range(cells[p].nleaf):
            l = cells[p].leaf[i]
            dx, dy, dz = cells[p].x-particles[l].x, cells[p].y-particles[l].y, cells[p].z-particles[l].z
            cells[p].multipole += particles[l].w * np.array((1, dx, dy, dz,\
                                               dx**2/2, dy**2/2, dz**2/2,\
                                               dx*dy/2, dy*dz/2, dz*dx/2)) 
        leaves.append(p)

# calculate parent's multilope based on its children's multipole
def get_parent_cell_multipole(p, c, cells):
    dx, dy, dz = cells[p].x-cells[c].x, cells[p].y-cells[c].y, cells[p].z-cells[c].z
    
    Dxyz =  np.array((dx, dy, dz))
    Dyzx = np.roll(Dxyz,-1) #It permutes the array (dx,dy,dz) to (dy,dz,dx) 
    
    cells[p].multipole += cells[c].multipole
    
    # dipole
    cells[p].multipole[1:4] += cells[c].multipole[0] * Dxyz
    
    # quadrop
    cells[p].multipole[4:7] += cells[c].multipole[1:4] * Dxyz\
                             + 0.5*cells[c].multipole[0] *  Dxyz**2
    
    cells[p].multipole[7:] += 0.5*np.roll(cells[c].multipole[1:4], -1) * Dxyz \
                            + 0.5*cells[c].multipole[1:4] * Dxyz \
                            + 0.5*cells[c].multipole[0] * Dxyz * Dyzx   

def upward_sweep(cells):
    for c in range(len(cells)-1, 0, -1):
        p = cells[c].parent
        get_parent_cell_multipole(p, c, cells)

def direct_sum(particles):
    u = np.zeros(len(particles))
    for i, target in enumerate(particles):
        for source in (particles[:i] + particles[i+1:]):
            r = target.distance(source)
            u[i] += source.w / r
    return u

def evaluate(particles, p, i, cells, sigma, theta):
    # non-leaf cell
    if cells[p].nleaf >= sigma:
        
        # loop in p's child cells (8 octants)
        for octant in range(8):
            if cells[p].clist & (1 << octant):
                c = cells[p].child[octant]
                r = particles[i].distance(cells[c])
                
                # near-field child cell
                if cells[c].r > theta*r:
                    evaluate(particles, c, i, cells, sigma, theta)
                
                # far-field child cell
                else:
                    dx = particles[i].x - cells[c].x
                    dy = particles[i].y - cells[c].y
                    dz = particles[i].z - cells[c].z
                    r3 = r**3
                    r5 = r3*r**2
                
                    # calculate the weight for each multipole
                    multipole_weight = [1/r, -dx/r3, -dy/r3, -dz/r3, 3*dx**2/r5 - 1/r3, \
                              3*dy**2/r5 - 1/r3, 3*dz**2/r5 - 1/r3, 3*dx*dy/r5, \
                              3*dy*dz/r5, 3*dz*dx/r5]
                    
                    particles[i].u += np.dot(cells[c].multipole, multipole_weight)
                
    # leaf cell
    else:
        # loop in twig cell's particles
        for l in range(cells[p].nleaf):
            source = particles[cells[p].leaf[l]]
            r = particles[i].distance(source)
            if r != 0:
                particles[i].u += source.w / r

def eval_potential(particles, cells, sigma, theta):
    for i in range(len(particles)):
        evaluate(particles, 0, i, cells, sigma, theta)

def l2_err(u_direct, u_tree):
    err = np.sqrt(sum((u_direct-u_tree)**2) / sum(u_direct**2))
    print('L2 Norm error: {}'.format(err))


def plot_err(u_direct, u_tree): 
    # plotting the relative error band
    n = len(u_direct)
    err_rel = abs((u_tree - u_direct) / u_direct)
    
    plt.figure(figsize=(10,4))
    ax = plt.gca()
    plt.plot(range(n), err_rel, 'bo', alpha=0.5)
    plt.xlim(0,n-1)
    plt.ylim(1e-6, 1e-1)
    ax.yaxis.grid()
    plt.xlabel('target particle index')
    plt.ylabel(r'$e_{\phi rel}$')
    ax.set_yscale('log')
    plt.show()


def plot_particles(particles):
    # plot spatial particle distribution
    fig = plt.figure(figsize=(15,4.5))
    
    # left plot
    # ax = fig.add_subplot(1,2,1, projection='3d')
    # ax.scatter([particle.x for particle in particles], 
               # [particle.y for particle in particles], 
               # [particle.z for particle in particles], s=30, c='b')
    # ax.set_xlim3d(0,1)
    # ax.set_ylim3d(0,1)
    # ax.set_zlim3d(0,1)
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$y$')
    # ax.set_zlabel(r'$z$')
    # ax.set_title('Particle Distribution')
    
    # right plot
    ax = fig.add_subplot(1,2,1, projection='3d')
    scale = 50   # scale for dot size in scatter plot
    u_list = []
    for particle in particles:
        u_list.append(particle.u)
    u_min, u_max = np.min(u_list), np.max(u_list)
    color_norm = plt.Normalize(u_min, u_max)
    cmap = plt.cm.get_cmap('viridis')
    scatter = ax.scatter([particle.x for particle in particles], 
               [particle.y for particle in particles], 
               [particle.z for particle in particles],
                s=10, c = u_list, cmap = cmap, norm = color_norm)
    ax.set_xlim3d(0,1)
    ax.set_ylim3d(0,1)
    ax.set_zlim3d(0,1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')
    ax.set_title('Particle Distribution (color implies potential)');

    # Add colorbar legend
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Potential')
    cbar.ax.set_position([0.3, 0.1, 0.03, 0.8])
    plt.subplots_adjust(right=0.8)
    plt.show()

def plot_potential(particles):
    # 提取粒子的坐标和电势值
    x = [particle.x for particle in particles]
    y = [particle.y for particle in particles]
    z = [particle.u for particle in particles]

    # 创建网格数据
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)  # 创建和 X 相同形状的全零数组
    for i in range(len(x)):
        Z[i, i] = z[i]

    # 创建3D子图并绘制光滑曲面
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_zlim3d(225, 425)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Electric Potential')

    # 显示图形
    plt.show()

def read_particle(filename):
    file = open('test/' + filename, 'r')
    particles = []
    for line in file:
        line = [float(x) for x in line.split()]
        coord, w = line[1:4], line[-1]
        particle = Particle(coord = coord, w = w)
        particles.append(particle)
    file.close()
    
    return particles

def write_result(u, filename):
    file = open('test/' + filename, 'w')
    for i in u:
        file.write(str(i) + '\n')
    file.close()


