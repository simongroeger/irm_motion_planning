import jax.numpy as jnp

class Environment:
    def __init__(self, link_length):
        self.link_length = link_length

    def fk(self, config):
        c2 = config.reshape(-1, 3)
        c = jnp.cumsum(c2, axis=1)
        pos_x = self.link_length @ jnp.cos(c).T
        pos_y = self.link_length @ jnp.sin(c).T
        pos = jnp.stack((pos_x, pos_y))
        return pos

    def compute_cost(self, f):
        # Example implementation of cost based on positions
        # Assuming cost is defined as some function of f (positions)
        # Return shape (t_len,)
        return jnp.sum(jnp.square(f), axis=0)  # Example: sum of squares of positions
    
    def compute_cost_vg(self, f):
        # Placeholder for gradient of cost_v w.r.t f
        # Assume cost_v = jnp.sum(jnp.square(f), axis=0)
        # Derivative w.r.t f is 2 * (f)
        return self.compute_cost(f), 2 * f

class CostFunction:
    def __init__(self, env):
        self.env = env

    def compute_point_cost(self, f, lambda_max_cost):
        cost_v = self.env.compute_cost(f)

        # Max and average costs
        max_cost = jnp.max(cost_v)
        avg_cost = jnp.sum(cost_v) / cost_v.shape[0]
        
        # Final cost combining max and avg
        cost = lambda_max_cost * max_cost + (1 - lambda_max_cost) * avg_cost
        return cost

    def compute_trajectory_obstacle_cost(self, trajectory, lambda_max_cost):
        f = self.env.fk(trajectory)
        cost = self.compute_point_cost(f, lambda_max_cost)
        return cost

    def compute_trajectory_obstacle_cost_grad(self, trajectory, lambda_max_cost):
        # Compute the forward kinematics
        f = self.env.fk(trajectory)
        
        # Get the cost and its gradient with respect to f
        cost_v, grad_cost_v = self.env.compute_cost_vg(f)  # Assuming we have this method
        
        # Calculate the index of the maximum cost in cost_v
        idx_max = jnp.argmax(cost_v)
        t_len = cost_v.shape[0]

        # Gradient w.r.t. cost_v (from max and average costs)
        grad_max_cost_v = jnp.zeros_like(cost_v)
        grad_max_cost_v = grad_max_cost_v.at[idx_max].set(1)  # Derivative of max function

        grad_avg_cost_v = jnp.ones_like(cost_v) / t_len  # Derivative of avg function
        
        # Combine gradients with respect to cost_v
        grad_cost_v_combined = (
            lambda_max_cost * grad_max_cost_v + (1 - lambda_max_cost) * grad_avg_cost_v
        )

        # Now we need to propagate this back to the trajectory
        # First, we need the gradient of f w.r.t trajectory
        # f is of shape (2, t_len) - assume link_length is of shape (2, 3)
        # The partial derivative of f w.r.t trajectory is based on the Jacobian of fk
        
        # Jacobian for forward kinematics (shape (2, t_len, 3))
        c2 = trajectory.reshape(-1, 3)
        c = jnp.cumsum(c2, axis=1)
        
        # Jacobian calculation
        jacobian_x = jnp.zeros((2, c2.shape[0], 3))
        jacobian_y = jnp.zeros((2, c2.shape[0], 3))

        # For each joint configuration, compute the derivatives
        for i in range(c2.shape[0]):
            for j in range(i + 1):
                jacobian_x.at[0, i].set(jacobian_x[0, i] - self.env.link_length[0] * jnp.sin(c[i, j]))
                jacobian_y.at[0, i].set(jacobian_y[0, i] + self.env.link_length[0] * jnp.cos(c[i, j]))
        
        for i in range(c2.shape[0]):
            for j in range(i + 1, c2.shape[0]):
                jacobian_x.at[1, j].set(jacobian_x[1, j] - self.env.link_length[1] * jnp.sin(c[j, i]))
                jacobian_y.at[1, j].set(jacobian_y[1, j] + self.env.link_length[1] * jnp.cos(c[j, i]))

        # Combine x and y Jacobians
        jacobian = jnp.stack((jacobian_x, jacobian_y), axis=1)  # shape (2, 2, t_len, 3)

        # Total gradient of cost w.r.t trajectory
        grad_trajectory = jnp.einsum('ij,jk->ik', grad_cost_v_combined, jacobian)
        
        return grad_trajectory

# Example usage
link_length = jnp.array([1.0, 1.0, 1.0])  # Example link lengths
env = Environment(link_length)
cost_func = CostFunction(env)

# Example input for trajectory (shape: t_len x 3)
t_len = 50
trajectory = jnp.array([[0.1, 0.2, 0.3]] * t_len)

lambda_max_cost = 0.7

# Compute the cost
cost = cost_func.compute_trajectory_obstacle_cost(trajectory, lambda_max_cost)
print("Cost:", cost)

# Compute the analytical gradient
grad_cost = cost_func.compute_trajectory_obstacle_cost_grad(trajectory, lambda_max_cost)
print("Gradient of the cost w.r.t. trajectory:", grad_cost)
