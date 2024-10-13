# irm_motion_planning

This project provides a trajectory optimization using the RKHS and functional-gradient descent.

## Starting the code
The code can be started with the main.py and different hyperparameter can easily bechanged using the argparser.

The project is divided into different python files:
main.py manages the hyperparameters and calls the optimization
There are 2 optimizers implemented: optimizer_BLS.py and optimizerGD.y
In trajectory.py the trajectory is defined and the robot is modeled in robot.py.
In environment.yp the environment is defined which can be dynamically changed during deployment without having to re-compile.

## Visualization
The visualization is in a seperate directory. To solve some issues with python's relative includes the visualization has to be started from this folder.
It can visualize the whole process, only the environment or create an animation of the robots movement or the optimization process.

## Blog
The blog can be ound in DevBlog-Theme in blog-post.html and can be opened by simply opening the file in a browser.
