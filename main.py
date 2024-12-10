'''
EECE Project
Smooth Path Planning as an Optimization Problem
Author: Ryan Huang

Run this file
'''

from RRT import RRT
from utility import extract_rectangle_vertices, plot_environment
import numpy as np
from astar import AStar

if __name__ == "__main__":
    # Define the basic environment
    start = (0, 0)
    goal = (50, 50)
    bounds = [(0, 60), (0, 60)]  # Extra padding to aid visualization
    num_obstacles = 20
    obstacles = []

    # Random Obstacle Generator
    while len(obstacles) < num_obstacles:
        location = np.random.randint(5, 50, (1, 2))
        size = np.random.randint(5, 10, (1, 2))
        obst = np.concatenate((location, size)).reshape(1, 4)

        if (obst[0][0] + obst[0][2] > goal[0] and obst[0][1] + obst[0][3] > goal[1]) or \
                (obst[0][0] < start[0] and obst[0][1] < start[1]):
            continue
        else:
            obstacles.append(tuple(obst.tolist()[0]))

    # Create RRT object
    rrt = RRT(start=start, goal=goal, bounds=bounds, obstacles=obstacles, step_size=2)
    rrt.plan()     # Plan the path
    path = rrt.get_path()     # Get the path from start to goal

    # Create an A* object and find the path
    a_star = AStar(start, goal, bounds, obstacles)
    a_star_path = a_star.search()

    # Get the vertices of the obstacles
    avoid_x, avoid_y = extract_rectangle_vertices(obstacles)

    # Shrink the number of spline points to optimize to speed up computation
    # cut_path = path[0::5]
    cut_path = a_star_path[0::2]
    if goal not in cut_path:
        cut_path = np.append(cut_path, goal)

    # Plot the environment and the resulting path
    plot_environment(rrt, a_star_path, [avoid_x, avoid_y], obstacles, path=cut_path, optimize=True)