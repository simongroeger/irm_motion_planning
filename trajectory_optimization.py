import numpy as np
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt
import torch
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation


N_timesteps = 15
N_joints = 3

obstacles = torch.tensor([
                        [ 2, -3],
                        [-2, 2],
                        [3, 3],
                        [-1, -2],
                        [-2, 1],
                        [-1, -1],
                        [-2, -3],
                        [-2, 0],
                        [1, 3],
                        [3, 2],
                        [2, 3]
                        ])


link_length = torch.tensor([1.0, 1.0, 1.0])

start_config = torch.tensor([0.0, 0.0, 0.0])
goal_config = torch.tensor([1.2, 0.8, 0.3])


straight_line = torch.stack((
    torch.linspace(start_config[0], goal_config[0], N_timesteps),
    torch.linspace(start_config[1], goal_config[1], N_timesteps),
    torch.linspace(start_config[2], goal_config[2], N_timesteps)    
)).T
straight_line.requires_grad = True

cart = torch.Tensor([[2.5, 0.0, 2.0],
                     [2.0, 1.0, 2.0]])
cart.requires_grad = True



def plot_loss_contour(fig, ax):

    # make these smaller to increase the resolution
    dx, dy = 0.05, 0.05

    # generate 2 2d grids for the x & y bounds
    x, y = np.mgrid[slice(-4, 4 + dy, dy),
                    slice(-4, 4 + dx, dx)]

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



# Creates the animation.
def create_animation(data):
    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Plot true hypothesis.
    plot_loss_contour(fig, ax)


    start_cart = fk(start_config).detach().numpy()
    plt.plot(start_cart[0], start_cart[1], 'o', color="yellow")

    goal_cart = fk(goal_config).detach().numpy()
    plt.plot(goal_cart[0], goal_cart[1], 'o', color="orange")

        


    cartesian_data = fk(data[0]).detach().numpy()
    curr_fx, = ax.plot(cartesian_data[0], cartesian_data[1], '-', c='tab:red', label='Learned Hypothesis')
    ax.legend(loc='lower center')

    # Title.
    title_text = 'Trajectory Optimization \n Iteration %d'
    title = plt.text(0.5, 0.85, title_text, horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)

    # Init only required for blitting to give a clean slate.
    def init():
        cartesian_data = fk(data[0]).detach().numpy()
        curr_fx.set_xdata(cartesian_data[0])
        curr_fx.set_ydata(cartesian_data[1])
        return curr_fx, title,

    # Update at each iteration.
    def animate(iteration):
        cartesian_data = fk(data[iteration]).detach().numpy()
        curr_fx.set_xdata(cartesian_data[0])
        curr_fx.set_ydata(cartesian_data[1])
        title.set_text(title_text % iteration)
        return curr_fx, title,

    ani = FuncAnimation(fig, animate, len(data.keys()), init_func=init, interval=25, blit=True, repeat=False)
    ani.save('to.gif', writer='imagemagick', fps=60)





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
    c2 = config.reshape(-1, 3)
    c = torch.cumsum(c2,dim=1)
    pos_x = torch.matmul(link_length, torch.cos(c).T)
    pos_y = torch.matmul(link_length, torch.sin(c).T)
    pos = torch.stack((pos_x, pos_y))
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

def compute_cartesian_cost(f):
    d = 2
    t_len = f.shape[1]
    o_len = obstacles.shape[0]
    a = f.reshape((d, t_len, 1)).expand((d, t_len, o_len))
    b1 = obstacles.T.reshape((d, 1, -1))
    b = b1.expand(d, t_len, o_len)
    cost_v = torch.sum(0.8 / (0.5 + torch.norm(a - b, dim=0)), dim=1)
    cost = torch.sum(cost_v)
    return cost, cost_v

def compute_raw_trajectory_obstacle_cost(trajectory):
    f = fk(trajectory)
    return compute_cartesian_cost(f)


def compute_trajectory_obstacle_cost(alpha, km):
    trajectory = evaluate(alpha, km)
    return compute_raw_trajectory_obstacle_cost(trajectory)


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



#plot_loss_contour()


t, alpha, km = init_trajectory()

max_iteration = 100
lr = 0.01         
lambda_reg = 0.02

#alpha.retain_grad()

data = {}


#for iter in range(max_iteration):
#    loss, _ = compute_trajectory_obstacle_cost(alpha, km)
#    loss.backward()
#    with torch.no_grad():
#        print(iter, loss.item())
#        alpha.data = alpha.data - lr * alpha.grad.data
#        alpha.grad.zero_()

data[0] = straight_line

for iter in range(200):
    loss, _ = compute_raw_trajectory_obstacle_cost(straight_line)
    loss.backward()
    with torch.no_grad():
        print(iter, loss.item())

        b = fk(straight_line)
        #plt.plot([b[0,8].item()], [b[1,8].item()], 'o', color="brown")

        straight_line.data = straight_line.data - lr * straight_line.grad.data
        straight_line.grad.zero_()

        data[iter + 1] = straight_line.detach().clone()





for iter in range(70):
    loss, li = compute_cartesian_cost(cart)
    loss.backward()
    with torch.no_grad():
        print()
        print(iter, loss.item())
        #print(cart[:, 0].detach().numpy(), li[0].detach().numpy(), compute_point_obstacle_cost(np.array([[cart[0, 0].item()]]), np.array([[cart[1, 0].item()]])))
        #print(cart.grad)

        #plt.plot([cart[0,1].item()], [cart[1,1].item()], 'o', color="brown")

        cart.data = cart.data - lr * cart.grad.data
        cart.grad.zero_()



create_animation(data)


exit(0)

start_cart = fk(start_config).detach().numpy()
plt.plot(start_cart[0], start_cart[1], 'o', color="yellow")

goal_cart = fk(goal_config).detach().numpy()
plt.plot(goal_cart[0], goal_cart[1], 'o', color="orange")

a = fk(evaluate(alpha, km)).detach().numpy()
plt.plot(a[0], a[1])

b = fk(straight_line).detach().numpy()
plt.plot(b[0], b[1])

c = cart.detach().numpy()
plt.plot(c[0], c[1])

plt.plot([0], [0], 'o', color="black")
plt.scatter(obstacles[:, 0], obstacles[:, 1])

plt.show()

    





