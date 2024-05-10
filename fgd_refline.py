"""
Functional Gradient Descent: A Toy Example.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Create toy dataset.
def create_dataset():
    size = 50
    data = np.loadtxt('refline.txt')
    t = []
    x = []
    y = []
    sum_t = 0
    
    for i in range(1,size):
        sum_t += np.linalg.norm(data[i-1,:2] - data[i,:2])
    max_t = sum_t
    sum_t = 0
    t.append(0)
    x.append(data[0,0])
    y.append(data[0,1])
    for i in range(1,size):
        sum_t += np.linalg.norm(data[i-1,:2] - data[i,:2])
        t.append(sum_t / max_t)
        x.append(data[i,0])
        y.append(data[i,1])

    return np.array(t), np.array(x), np.array(y)


# The kernel we use.
def poly_kernel(x_1, x_2):
    return np.exp(-(((x_1 - x_2)/0.08)**2))


# Create kernel matrix from dataset.
def create_kernel_matrix(x, kernel_f):
    kernel_matrix = np.zeros((x.size, x.size))
    for i, x_i in enumerate(x):
        for j, x_j in enumerate(x):
            kernel_matrix[i][j] = kernel_f(x_i, x_j)
    return kernel_matrix


# Evaluate f(x) = [f(x_1), ..., f(x_n)] with coefficients alpha.
def evaluate(alpha, kernel_matrix):
    return np.matmul(kernel_matrix, alpha)


def eval_any(alpha, kernel_f, support_x, eval_x):
    kernel_matrix = np.zeros((eval_x.size, support_x.size))
    for i, x_i in enumerate(eval_x):
        for j, x_j in enumerate(support_x):
            kernel_matrix[i][j] = kernel_f(x_i, x_j)
    return np.matmul(kernel_matrix, alpha)


# Creates the animation.
def create_animation(t, x, y, fxs, fys):
    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # Plot true hypothesis.
    ax.plot(t, y, '-o', c='tab:blue', label='True Hypothesis')
    ax.set_xlim(np.min(t) - 0.1, np.max(t) + 0.1)
    ax.set_ylim(np.min(y) - 0.1, np.max(y) + 0.1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    curr_fx, = ax.plot(t, fys[0], '-', c='tab:red', label='Learned Hypothesis')
    ax.legend(loc='lower center')

    #for i in range(100):
    #    print(i)
    #    plt.draw()
    #    plt.pause(0.1)
    #    #curr_fx, = ax.plot(fxs[i], fys[i], '-', c='tab:red', label='Learned Hypothesis')
    #    curr_fx.set_ydata(np.ma.array(fys[i], mask=True))
    #    #curr_fx.set_xdata(np.ma.array(fxs[i], mask=True))

    #plt.show()
    #return

    # Title.
    title_text = 'Functional Gradient Descent \n Iteration %d'
    title = plt.text(0.5, 0.85, title_text, horizontalalignment='center', verticalalignment='center', transform=fig.transFigure, fontsize=14)

    # Init only required for blitting to give a clean slate.
    def init():
        curr_fx.set_ydata(np.ma.array(fys[iteration], mask=True))
        curr_fx.set_xdata(np.ma.array(t, mask=True))
        return curr_fx, title,

    # Update at each iteration.
    def animate(iteration):
        curr_fx.set_ydata(fys[iteration])
        curr_fx.set_xdata(t)
        title.set_text(title_text % iteration)
        return curr_fx, title,

    ani = FuncAnimation(fig, animate, len(fxs.keys()), init_func=init, interval=25, blit=True, repeat=False)
    ani.save('fgd_refline.gif', writer='imagemagick', fps=60)


if __name__ == "__main__":

    # Global constants.
    lr = 0.01              # Learning rate.
    lambda_reg = 0.1        # Regularization coefficient.
    num_iterations = 500   # Iterations for gradient descent.
    seed = 0                # Random seed.

    # Seed for reproducibility.
    np.random.seed(seed)

    # Obtain data.
    t, x, y = create_dataset()
    kernel_matrix = create_kernel_matrix(t, poly_kernel)

    x_eval = np.linspace(-1, 1, 40)
    y_eval = np.linspace(-1, 1, 40)

    # Store fx (training set evaluations) at each iteration. We will use this to create an animation.
    fxs = {}
    fys = {}

    # Initialize and iterate.
    alpha_x = np.random.randn(t.size)
    alpha_y = np.random.randn(t.size)
    start = time.time()
    for iteration in range(num_iterations):
        
        # Evaluate fx using the current alpha.
        fx = evaluate(alpha_x, kernel_matrix)
        fy = evaluate(alpha_y, kernel_matrix)

        #y_eval = eval_any(alpha, poly_kernel, x, x_eval)
        
        # Save.
        fxs[iteration] = fx
        fys[iteration] = fy

        # Compute loss (just for logging!).
        loss_x = np.sum(np.square(x - fx)) + lambda_reg * (np.matmul(alpha_x, fx))
        loss_y = np.sum(np.square(y - fy)) + lambda_reg * (np.matmul(alpha_y, fy))
        print('Iteration %d: Loss = %0.3f %0.3f' % (iteration, loss_x, loss_y))

        # Compute gradient and update.
        alpha_x = 2 * lr * (x - fx) + (1 - 2 * lambda_reg * lr) * alpha_x
        alpha_y = 2 * lr * (y - fy) + (1 - 2 * lambda_reg * lr) * alpha_y

    end = time.time()
    print("took", end-start, "s")


    create_animation(t, y, x, fys, fxs)

