import numpy as np
import matplotlib.pyplot as plt
from math import atan2
import torch

track = torch.from_numpy(np.loadtxt("motion_planning.txt")[:,:2])


def init_trajectory(track):
    trajectory = track.clone()
    trajectory.requires_grad = True
    return trajectory


def obstacle_cost(trajectory):
    res = torch.sum(torch.norm(trajectory - track))
    return res

def smoothness_cost(trajectory):
    res = torch.tensor(0.0, requires_grad=True)
    for i in range(2, len(trajectory)):
        yaw_prev = torch.atan2(trajectory[i-1,1] - trajectory[i-2,1], trajectory[i-1,0] - trajectory[i-2,0])
        yaw = torch.atan2(trajectory[i,1] - trajectory[i-1,1], trajectory[i,0] - trajectory[i-1,0])
        yaw_diff = torch.abs(yaw - yaw_prev)
        if yaw_diff > 2 * torch.pi:
            yaw_diff -= 2 * torch.pi
        distance = 0.5 * (torch.norm(trajectory[i-1] - trajectory[i-2]) + torch.norm(trajectory[i] - trajectory[i-1]))
        res = res + yaw_diff / distance
    return res

def cost(trajectory):
    w = 0.1
    return w * obstacle_cost(trajectory) + (1-w) * smoothness_cost(trajectory)



trajectory = init_trajectory(track)

learning_rate = 0.001

best_trajectory = trajectory.clone()

for iter in range(1000):
    loss = cost(trajectory)
    loss.backward()
    with torch.no_grad():
        if loss < cost(best_trajectory):
            best_trajectory = trajectory.clone()
        print(loss.item())
        #print(trajectory.grad)
        trajectory -= learning_rate * trajectory.grad

    

tnp = best_trajectory.detach().numpy()

plt.plot(tnp[:, 1], tnp[:, 0], label="trajectory")
plt.plot(track[:, 1], track[:, 0], label="track")
plt.legend()
plt.show()



print(track.shape)