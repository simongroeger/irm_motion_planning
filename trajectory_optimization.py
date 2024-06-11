import numpy as np
import matplotlib.pyplot as plt
from math import atan2, sin, cos, sqrt
import torch
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation

#param
N_timesteps = 100
N_joints = 3

rbf_var = 0.2

max_iteration = 800
lr_start = 0.001
lr_end =   0.000001
lambda_reg = 0.1
lambda_constraint = 2

def lr(iter):
    return lr_start + (lr_end - lr_start) * iter / max_iteration


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


link_length = torch.tensor([1.5, 1.0, 0.5])

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


custom_trajectory = torch.Tensor([[0.0, 0.0, 0.0],
                                  [0.0, -0.4, 0.15],
                                  [0.0, 0.8, 0.3],
                                  [0.4, 0.8, 0.3],
                                  [0.8, 0.8, 0.3],
                                  [1.2, 0.8, 0.3]])



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

    #draw
    levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())

    cf = ax.contourf(x[:-1, :-1] + dx/2.,
                    y[:-1, :-1] + dy/2., z, levels=levels,
                    cmap=plt.get_cmap('PiYG'))
    fig.colorbar(cf, ax=ax)
    
    fig.tight_layout()



# Creates the animation.
def create_animation(data, losses):
    
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
    
    # Plot true hypothesis.
    plot_loss_contour(fig, ax0)
    plot_loss_contour(fig, ax2)

    #plot start and goal
    start_cart = fk(start_config).detach().numpy()
    ax0.plot(start_cart[0], start_cart[1], 'o', color="yellow", label="start_config")
    ax2.plot(start_cart[0], start_cart[1], 'o', color="yellow", label="start_config")

    goal_cart = fk(goal_config).detach().numpy()
    ax0.plot(goal_cart[0], goal_cart[1], 'o', color="gold", label="goal_config")
    ax2.plot(goal_cart[0], goal_cart[1], 'o', color="gold", label="goal_config")

    ax0.plot([0], [0], 'o', color="black", label="robot base = joint 0")
    ax2.plot([0], [0], 'o', color="black", label="robot base = joint 0")

    # plot workspace
    #circle1 = plt.Circle((0, 0), 3, color='black', fill=False)
    #ax0.add_patch(circle1)

    # plot init trajectory
    #cartesian_data = fk(data[0]).detach().numpy()
    #ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='tab:brown')
    cartesian_data = fk(straight_line).detach().numpy()
    ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='tab:gray', label="initial straight line trajectory")

    #cartesian_data = fk(custom_trajectory).detach().numpy()
    #ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='tab:green')


    cartesian_data = fk_joint(data[0], 1).detach().numpy()
    curr_fx1, = ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='blue', label="joint 1")
    cartesian_data = fk_joint(data[0], 2).detach().numpy()
    curr_fx2, = ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='orange', label="joint 2")
    cartesian_data = fk(data[0]).detach().numpy()
    curr_fx, = ax0.plot(cartesian_data[0], cartesian_data[1], '-', c='darkgreen', label="ee")
    

    ax0.legend(loc='lower center')

    ax1.plot(t, straight_line[:, 0].detach().numpy(), '-', color='darkblue')
    ax1.plot(t, straight_line[:, 1].detach().numpy(), '-', color='orangered')
    ax1.plot(t, straight_line[:, 2].detach().numpy(), '-', color='darkgreen')

    ax1.plot(t, data[len(data)-1][:, 0].detach().numpy(), '-', color='grey')
    ax1.plot(t, data[len(data)-1][:, 1].detach().numpy(), '-', color='grey')
    ax1.plot(t, data[len(data)-1][:, 2].detach().numpy(), '-', color='grey')

    s1, = ax1.plot(t, data[0][:, 0].detach().numpy(), '-', color='blue', label="joint 0 angle")
    s2, = ax1.plot(t, data[0][:, 1].detach().numpy(), '-', color='orange', label="joint 1 angle")
    s3, = ax1.plot(t, data[0][:, 2].detach().numpy(), '-', color='green', label="joint 2 angle")


    last_id = len(data.keys())-1 
    fin_movement = np.zeros((4, 2, N_timesteps))
    fin_movement[1] = fk_joint(data[last_id], 1).detach().numpy()
    fin_movement[2] = fk_joint(data[last_id], 2).detach().numpy()
    fin_movement[3] = fk_joint(data[last_id], 3).detach().numpy()
    s4,  = ax2.plot(fin_movement[:,0,0], fin_movement[:,1,0], '-', color = 'black', label="robot")

    # Title.
    title_text = 'Trajectory Optimization \n Iteration %d, Loss %.2f'
    title = ax1.text(0.5, 0.4, title_text, horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)

    # Init only required for blitting to give a clean slate.
    def init():
        cartesian_data = fk(data[0]).detach().numpy()
        curr_fx.set_xdata(cartesian_data[0])
        curr_fx.set_ydata(cartesian_data[1])

        cartesian_data = fk_joint(data[0],1).detach().numpy()
        curr_fx1.set_xdata(cartesian_data[0])
        curr_fx1.set_ydata(cartesian_data[1])

        cartesian_data = fk_joint(data[0],2).detach().numpy()
        curr_fx2.set_xdata(cartesian_data[0])
        curr_fx2.set_ydata(cartesian_data[1])

        s1.set_ydata(data[0][:, 0].detach().numpy())
        s2.set_ydata(data[0][:, 1].detach().numpy())
        s3.set_ydata(data[0][:, 2].detach().numpy())

        title.set_text(title_text % (0, 0))
        return curr_fx, curr_fx1, curr_fx2, s1, s2, s3, s4, title,

    # Update at each iteration.
    def animate(iteration):
        if iteration % 10 == 0:
            cartesian_data = fk(data[iteration]).detach().numpy()
            curr_fx.set_xdata(cartesian_data[0])
            curr_fx.set_ydata(cartesian_data[1])

            cartesian_data = fk_joint(data[iteration],1).detach().numpy()
            curr_fx1.set_xdata(cartesian_data[0])
            curr_fx1.set_ydata(cartesian_data[1])

            cartesian_data = fk_joint(data[iteration],2).detach().numpy()
            curr_fx2.set_xdata(cartesian_data[0])
            curr_fx2.set_ydata(cartesian_data[1])

            s1.set_ydata(data[iteration][:, 0].detach().numpy())
            s2.set_ydata(data[iteration][:, 1].detach().numpy())
            s3.set_ydata(data[iteration][:, 2].detach().numpy())
            title.set_text(title_text % (iteration, losses[iteration]))
        
        ri = int(iteration * N_timesteps / len(data.keys()))
        s4.set_xdata(fin_movement[:,0,ri])
        s4.set_ydata(fin_movement[:,1,ri])

        return curr_fx, curr_fx1, curr_fx2, s1, s2, s3, s4, title,

    ani = FuncAnimation(fig, animate, len(data.keys()), init_func=init, interval=20, blit=True, repeat=False)
    plt.show()
    #ani.save('to.gif', writer='imagemagick', fps=60)





def jacobian(config):
    c2 = config.reshape(-1, 3)
    c = torch.cumsum(c2,dim=1)
    
    x1 = - torch.matmul(link_length, torch.sin(c).T)
    x2 = - torch.matmul(link_length[1:], torch.sin(c[:, 1:]).T)
    x3 = - torch.matmul(link_length[2:], torch.sin(c[:, 2:]).T)
    y1 = torch.matmul(link_length, torch.cos(c).T)
    y2 = torch.matmul(link_length[1:], torch.cos(c[:, 1:]).T)
    y3 = torch.matmul(link_length[2:], torch.cos(c[:, 2:]).T)
    
    j = torch.Tensor([[x1, x2, x3], [y1, y2, y3]])
    return j



def rbf_kernel(x_1, x_2):
    return torch.exp( - (x_1 - x_2)**2 / (2 * rbf_var**2) )


# Create kernel matrix from dataset.
def create_kernel_matrix(kernel_f, x, x2):
    kernel_matrix = torch.zeros((x.shape[0], x2.shape[0]))
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x2):
            kernel_matrix[i][j] = kernel_f(x_i, x_j)    
    return kernel_matrix

def evaluate(alpha, kernel_matrix, jac):
    return kernel_matrix @ alpha @ jac

def eval_any(alpha, kernel_f, support_x, eval_x, jac):
    return evaluate(alpha, create_kernel_matrix(kernel_f, eval_x, support_x), jac)

def eval_any_single(alpha, kernel_f, support_x, eval_x, jac):
    return kernel_f(support_x, eval_x) @ jac @ alpha  


def fk(config):
    c2 = config.reshape(-1, 3)
    c = torch.cumsum(c2,dim=1)
    pos_x = link_length @ torch.cos(c).T
    pos_y = link_length @ torch.sin(c).T
    pos = torch.stack((pos_x, pos_y))
    return pos


def fk_joint(config, joint_id):
    c2 = config[:, :joint_id].reshape(-1, joint_id)
    c = torch.cumsum(c2,dim=1)
    ll = link_length[:joint_id]
    pos_x = ll @ torch.cos(c).T
    pos_y = ll @ torch.sin(c).T
    pos = torch.stack((pos_x, pos_y))
    return pos


def init_trajectory():
    t = torch.linspace(0, 1, N_timesteps)
    kernel_matrix = create_kernel_matrix(rbf_kernel, t, t)
    alpha = torch.randn((N_timesteps, N_joints), requires_grad=True)

    jc = 0.5 * (start_config + goal_config)
    a = jacobian(jc)
    unjac = jacobian(jc).T @ jacobian(jc)
    jac = unjac / torch.mean(unjac)
    #jac = torch.eye(3)

    lr = 0.002
    lambda_reg = 0.02

    #fit_trajectory_to_straigth_line
    with torch.no_grad():
        y = straight_line.clone()

        for iteration in range(500):

            # Evaluate fx using the current alpha.
            fx = evaluate(alpha, kernel_matrix, jac)

            # Compute loss (just for logging!).
            loss = torch.sum(torch.square(y - fx)) + lambda_reg * torch.sum((torch.matmul(alpha.T, fx)))
            print('Init %d: Loss = %0.3f' % (iteration, loss))

            # Compute gradient and update.
            alpha = 2 * lr * (y - fx) + (1 - 2 * lambda_reg * lr) * alpha

    alpha.requires_grad = True
    return t, alpha, kernel_matrix, jac



def compute_point_obstacle_cost(x,y):
    cost = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            point = np.array([x[i,j], y[i,j]])
            c = 0
            for o in obstacles.detach().numpy():
                c += 0.8  / (0.5 + np.linalg.norm(point - o))
            cost[i,j] = c
    return cost



def compute_cartesian_cost(f):
    d = 2
    t_len = f.shape[1]
    o_len = obstacles.shape[0]
    a = f.reshape((d, t_len, 1)).expand((d, t_len, o_len))
    b1 = obstacles.T.reshape((d, 1, -1))
    b = b1.expand(d, t_len, o_len)
    cost_v = torch.sum(0.8 / (0.5 + torch.norm(a - b, dim=0)), dim=1)
    #cost = torch.sum(cost_v) / N_timesteps
    cost = torch.max(cost_v) + torch.sum(cost_v) / N_timesteps
    return cost

def compute_raw_trajectory_obstacle_cost(trajectory):
    f = fk(trajectory)
    return compute_cartesian_cost(f)


def compute_trajectory_obstacle_cost(alpha, km, jac):
    trajectory = evaluate(alpha, km, jac)
    return compute_raw_trajectory_obstacle_cost(trajectory)


def start_goal_cost(alpha, km, jac):
    trajectory = evaluate(alpha, km, jac)
    s = trajectory[0]
    g = trajectory[N_timesteps-1]
    loss = torch.norm(s-start_config) + torch.norm(g-goal_config)
    return loss


def joint_limit_cost(alpha, km, jac):
    trajectory = evaluate(alpha, km, jac)
    loss = torch.sum(torch.exp(100*(trajectory - 1.5)) + torch.exp(100*(-trajectory - 0.5))) / N_timesteps
    return loss


def compute_trajectory_cost(alpha, km, jac):
    #print(compute_trajectory_obstacle_cost(alpha, km).item(), start_goal_cost(alpha, km).item(), joint_limit_cost(alpha, km).item())
    return compute_trajectory_obstacle_cost(alpha, km, jac) + lambda_constraint * start_goal_cost(alpha, km, jac) #+ lambda_constraint * joint_limit_cost(alpha, km, jac)






t, alpha, km, jac = init_trajectory()

data = {}
losses = {}

data[0] = evaluate(alpha, km, jac)
losses[0] = 0




for iter in range(max_iteration):
    trajectory = evaluate(alpha, km, jac)
    loss = compute_trajectory_cost(alpha, km, jac)
    loss.backward()
    with torch.no_grad():
        if iter % 10 == 0: print(iter, loss.item())
        alpha.data = (1 - lambda_reg * lr(iter)) * alpha.data - lr(iter) * alpha.grad.data
        #alpha.data = alpha.data - lr(iter) * alpha.grad.data
        alpha.grad.zero_()

        data[iter+1] = evaluate(alpha, km, jac)
        losses[iter+1] = loss.detach().item()

"""
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
"""



"""
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
"""


create_animation(data, losses)


exit(0)

start_cart = fk(start_config).detach().numpy()
plt.plot(start_cart[0], start_cart[1], 'o', color="yellow")

goal_cart = fk(goal_config).detach().numpy()
plt.plot(goal_cart[0], goal_cart[1], 'o', color="orange")

a = fk(evaluate(alpha, km, jac)).detach().numpy()
plt.plot(a[0], a[1])

b = fk(straight_line).detach().numpy()
plt.plot(b[0], b[1])

c = cart.detach().numpy()
plt.plot(c[0], c[1])

plt.plot([0], [0], 'o', color="black")
plt.scatter(obstacles[:, 0], obstacles[:, 1])

plt.show()

    





