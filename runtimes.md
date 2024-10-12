# Runtime in ms



Measurment 10 * 100

Jit
BLS: 3.12 ms, 0.01
GD: 7.26 ms, 0.06

NoJit:
BLS: 34.81 ms, 1.03
GD: 122.59 ms, 1.31

Gradient
analytical: 3.12 ms, 0.01
automatic: 3.56, 0.03 
nojit automatic: 292.66, 2.40







# Old

## Analytical Gradient

| CPU                               | Jitted Loop | Not Jitted    |
| --------------------------------- | ----------- | ------------- |
| Gradient Descent                  | 180         | 27.0          |
| Gradient Descent EarlyStopping    | 200         | 36.2          |
| Gradient Descent Dual Loop        | -           | 57.3          |
| Backtracking Line Search          | 476         | 105.7         |

| GPU                               | Jitted Loop | Not Jitted    |
| --------------------------------- | ----------- | ------------- |
| Gradient Descent                  | -           | -             |
| Gradient Descent EarlyStopping    | -           | -             |
| Backtracking Line Search          | -           | -             |



## Jax Gradient

| CPU                               | Jitted Loop | Not Jitted    |
| --------------------------------- | ----------- | ------------- |
| Theoretic MAX                     | 65.5        | 161.4         |
| Gradient Descent                  | 292.8       | 282.9         |
| Gradient Descent EarlyStopping    | 428.9       | 402.0         |
| Backtracking Line Search          | 766.0       | 714.9         |

| GPU                               | Jitted Loop | Not Jitted    |
| --------------------------------- | ----------- | ------------- |
| Theoretic MAX                     | 150.7       | 342.8         |
| Gradient Descent                  | 522.5       | 650.7         |
| Gradient Descent EarlyStopping    | 631.3       | 907.4         |
| Backtracking Line Search          | 1183.3      | 1983.2        |


