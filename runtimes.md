# Runtime in ms

## Analytical Gradient

| CPU                               | Jitted Loop | Not Jitted    |
| --------------------------------- | ----------- | ------------- |
| Gradient Descent                  | 180         | 27.0          |
| Gradient Descent EarlyStopping    | 200         | 36.2          |
| Gradient Descent Dual Loop        | -           | 65.3          |
| Backtracking Line Search          | 476         | 109.4         |

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


