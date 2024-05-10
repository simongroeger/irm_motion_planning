import numpy as np
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt
import torch
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


N_timesteps = 10
N_joints = 3

obstacles = torch.tensor([
                        [ 2, -3],
                        [-2, 2],
                        [3, 3],
                        [-1, -2],
                        [-2, 1],
                        [-1, -1],
                        [0, -3],
                        [-2, 0],
                        [1, 3],
                        [3, 0],
                        [3, 2],
                        [2, 3]
                        ])

link_length = torch.tensor([1.0, 1.0, 1.0])

start_config = torch.tensor([0.0, 0.0, 0.0])
goal_config = torch.tensor([1.0, 0.7, 0.3])



def poly_kernel(x_1, x_2):
    return np.exp(-(((x_1 - x_2)/0.5)**2))

# Create kernel matrix from dataset.
def create_kernel_matrix(kernel_f, x, x2):
    kernel_matrix = torch.zeros((x.shape[0], x2.shape[0]))
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x2):
            kernel_matrix[i][j] = kernel_f(x_i, x_j)
    return kernel_matrix

def evaluate(alpha, kernel_matrix):
    return torch.matmul(kernel_matrix, alpha)

def eval_any(alpha, kernel_f, support_x, eval_x):
    return evaluate(alpha, create_kernel_matrix(kernel_f, eval_x, support_x))
    

def fk(config):
    pos_x = torch.matmul(link_length, torch.cos(config).T)
    pos_y = torch.matmul(link_length, torch.sin(config).T)
    pos = torch.stack((pos_x, pos_y))
    return pos

    angle = torch.tensor(0.0, requires_grad=True)
    result = torch.zeros((N_joints, 2))
    for i in range(N_joints):
        #config somehow 1dim not 3dim
        angle = angle + config[i]
        pos[0] = pos[0] + link_length[i] * torch.cos(angle)
        pos[1] = pos[1] + link_length[i] * torch.sin(angle)
        result[i] = pos
    return pos


def init_trajectory():
    t = torch.linspace(0, 1, N_timesteps)
    kernel_matrix = create_kernel_matrix(poly_kernel, t, t)
    alpha = torch.randn((N_timesteps, N_joints), requires_grad=True)
    return t, alpha, kernel_matrix

    #trajectory = torch.rand((N_timesteps, N_joints), requires_grad=True)
    #for i in range(N_timesteps):
    #    trajectory.data[i] = start_config + (goal_config - start_config) * i / (N_timesteps - 1) + torch.rand(3)/20
    #return trajectory


"""
def compute_trajectory_smoothness_cost(trajectory):
    cost = torch.tensor(0.0, requires_grad=True)
    for i in range(1,len(trajectory)):
        cost = cost + torch.sum(torch.norm(trajectory[i] - trajectory[i-1]))
    return cost


def compute_trajectory_cost(trajectory):
    w = 1
    return compute_trajectory_obstacle_cost(trajectory) #+ (1-w) * compute_trajectory_smoothness_cost(trajectory)
"""


def compute_trajectory_obstacle_cost(alpha, km):
    trajectory = evaluate(alpha, km)
    f = fk(trajectory)
    d = 2
    t_len = f.shape[1]
    o_len = obstacles.shape[0]
    a = f.reshape((d, t_len, 1)).expand((d, t_len, o_len))
    b = obstacles.reshape((d, 1, -1)).expand(d, t_len, o_len)
    cost = torch.sum(5.0 / (1 + torch.norm(a - b, dim=0)))
    return cost

def compute_point_obstacle_cost(x,y):
    cost = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = np.array([x[i,j], y[i,j]])
            c = 0
            for o in obstacles.detach().numpy():
                c += 0.8 / (0.5 + np.linalg.norm(point - o))
            cost[i,j] = c
    return cost

t, alpha, km = init_trajectory()

max_iteration = 100
lr = 0.01         
lambda_reg = 0.02

#alpha.retain_grad()

for iter in range(max_iteration):
    loss = compute_trajectory_obstacle_cost(alpha, km)
    loss.backward()
    with torch.no_grad():
        print(iter, loss.item())
        alpha.data = alpha.data - lr * alpha.grad.data



def plot_loss_contour():

    # make these smaller to increase the resolution
    dx, dy = 0.05, 0.05

    # generate 2 2d grids for the x & y bounds
    y, x = np.mgrid[slice(-3, 3 + dy, dy),
                    slice(-3, 3 + dx, dx)]

    #z = np.sin(x)**10 + np.cos(10 + y*x) * np.cos(x)
    z = compute_point_obstacle_cost(x,y)

    # x and y are bounds, so z should be the value *inside* those bounds.
    # Therefore, remove the last value from the z array.
    z = z[:-1, :-1]
    levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())


    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    cmap = plt.get_cmap('PiYG')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig, ax = plt.subplots(nrows=1)

    #im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    #fig.colorbar(im, ax=ax0)
    #ax0.set_title('pcolormesh with levels')


    # contours are *point* based plots, so convert our bound into point
    # centers
    cf = ax.contourf(x[:-1, :-1] + dx/2.,
                    y[:-1, :-1] + dy/2., z, levels=levels,
                    cmap=cmap)
    fig.colorbar(cf, ax=ax)
    ax.set_title('contourf with levels')

    # adjust spacing between subplots so `ax1` title and `ax0` tick labels
    # don't overlap
    fig.tight_layout()


plt.plot([0], [0], 'o', color="black")
a = fk(evaluate(alpha, km)).detach().numpy()
plt.plot(a[0], a[1])

plt.scatter(obstacles[:, 0], obstacles[:, 1])


plt.show()

    





