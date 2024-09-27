import jax.numpy as jnp
from jax import random
key = random.PRNGKey(758493)  # Random seed is explicit in JAX


class Evaluator:
    def evaluate(self, alpha, kernel_matrix, jac):
        """Compute the evaluation of the product K * A * J"""
        return kernel_matrix @ alpha @ jac

    def compute_gradient(self, alpha, kernel_matrix, jac):
        """Compute the analytical gradient of the evaluate function with respect to alpha"""
        # Gradient with respect to alpha
        gradient = kernel_matrix.T @ jac  # (50, 50)^T @ (3, 3) = (50, 3)
        return gradient

# Example usage
random.uniform(key, shape=(1000,))

kernel_matrix = random.uniform(key, shape=(50, 50) ) # Random kernel matrix of shape (50, 50)
alpha = random.uniform(key, shape=(50, 3) )           # Random alpha of shape (50, 3)
jac = random.uniform(key, shape=(3, 3)   )             # Random jacobian of shape (3, 3)

evaluator = Evaluator()

# Compute the output
output = evaluator.evaluate(alpha, kernel_matrix, jac)
print("Output:\n", output)

# Compute the gradient with respect to alpha
gradient = evaluator.compute_gradient(alpha, kernel_matrix, jac)
print("Gradient with respect to alpha:\n", gradient)
