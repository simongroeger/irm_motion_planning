# irm_motion_planning

This project provides a trajectory optimization using the RKHS and functional-gradient descent.

## Installation
The required libraries can be installed using pip and was tested in Python3.8 in Ubuntu20
```
pip install -r requirements.txt
```

## Starting the code
The trajectory optimization code can be started with the main.py and different hyperparameter can be changed easily using the argparser.

The project is divided into different python files:
main.py manages the hyperparameters and calls the optimization

There are 2 optimizers implemented: optimizer_BLS.py and optimizerGD.py
<ul>
  <li>A Backtracking Line Search Gradient Descent is implemented in optimizer_BLS.py</li>
  <li>A regular Gradient Descent is implemented in optimizer_BLS.py</li>  
</ul>

In trajectory.py the trajectory is defined and the robot is modeled in robot.py.

In environment.py the environment is defined which can be dynamically changed during deployment without having to re-compile.

## Visualization
We split the visualization from the optimization itself and put is in a seperate directory.
To solve some issues with python's relative includes the visualization has to be started from the visualization folder.
The visualization loads a trajectory_result.txt or a trajectory_series.txt that was saved in the optimization process. 
If the main.py is started from within the visulaization folder, which is supported, then the files are already correctly saved in the visualization folder.
<ul>
  <li> visualization.py visualizes the whole process
  <li> env_vis.py visulaizes only the environment 
  <li> visualize_robotmovement.py create an animation of the robots movement
  <li> visualize_series.py creates an animation of the optimization process.
</ul>

## Blog
The blog can be found in DevBlog-Theme in blog-post.html and simply opening the file in a browser is sufficient to open the whole blog.
