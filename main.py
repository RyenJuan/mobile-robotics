'''
EECE Project
Smooth Path Planning as an Optimization Problem
Author: Ryan Huang

Run this file
'''

from RRT import RRT
from utility import extract_rectangle_vertices, plot_environment
import numpy as np

if __name__ == "__main__":
    # Define the basic environment
    start = (0, 0)
    goal = (50, 50)
    bounds = [(0, 60), (0, 60)]  # Extra padding to aid visualization
    num_obstacles = 5
    obstacles = []

    # Random Obstacle Generator
    for i in range(num_obstacles):
        location = np.random.randint(10, 40, (1, 2))
        size = np.random.randint(5, 20, (1, 2))
        obst = np.concatenate((location, size)).reshape(1, 4)
        obstacles.append(tuple(obst.tolist()[0]))

    # Create RRT object
    rrt = RRT(start=start, goal=goal, bounds=bounds, obstacles=obstacles, step_size=2)

    # Plan the path
    rrt.plan()

    # Get the path from start to goal
    path = rrt.get_path()

    # Get the vertices of the obstacles
    avoid_x, avoid_y = extract_rectangle_vertices(obstacles)

    # Shrink the number of spline points to optimize to speed up computation
    # FIXME: This doesn't actually work. The goal node is sometimes cut off
    #        Very easy fix, but I'm leaving this as a note to remind me to fix it
    cut_path = path[0::2]
    if goal not in cut_path:
        np.append(cut_path, goal)

    # Plot the environment and the resulting path
    plot_environment(rrt, [avoid_x, avoid_y], path=cut_path, optimize=True)
