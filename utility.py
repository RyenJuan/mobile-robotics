'''
EECE Project
Smooth Path Planning as an Optimization Problem
Author: Ryan Huang

Utility functions to clear up the global namespace of other files
'''

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
from optimizer import path_optimizer
from matplotlib.patches import Rectangle
from path_time import data_restructuring, time_calculator

def calculate_spline_length(cs_x, cs_y, t_start, t_end):
    """
    Calculate the arc length of a spline defined by cs_x and cs_y.

    :param cs_x: CubicSpline objects for x(t)
    :param cs_y: CubicSpline objects for y(t)
    :param t_start: Start of the spline parameter
    :param t_end: End of the spline parameter
    :return: Total arc length of the spline (float)
    """
    def arc_length_integrand(t):
        dx_dt = cs_x.derivative()(t)
        dy_dt = cs_y.derivative()(t)
        return np.sqrt(dx_dt**2 + dy_dt**2)

    # TODO: This method generates an integration warning basically everytime it runs
    #       Maybe investigate a more robust integration method
    length, _ = quad(arc_length_integrand, t_start, t_end)
    return length

def generate_vertical_points(num_points=100, vertical_spread=10, horizontal_spread=2):
    """
    Generate points mostly aligned along the vertical axis with some horizontal variation.

    :param num_points: Number of points to generate (int)
    :param vertical_spread: Spread of points along the y-axis (float)
    :param horizontal_spread: Spread of points along the x-axis (float)
    :return: 2d array of (x,y) coordinates (ndarray)
    """
    y_coords = np.linspace(0, 2*vertical_spread, num_points)
    x_coords = np.random.uniform(0, 2*horizontal_spread, num_points)

    points = np.column_stack((x_coords, y_coords))
    return points


def generate_cubic_spline(points, num_interpolated_points=200):
    """
    Generate a cubic spline spline through the given points.

    :param points: 2d array of (x,y) points to form the spline around (ndarray)
    :param num_interpolated_points: Number of points to form a smooth curve (int)
    :return: Tuple of ((x,y) interpolated points, cubic splines for x, cubic spline for y, parameter t)
    """
    # Calculate the chordal parameter t
    t = [0]
    for i in range(1, len(points)):
        chord_length = np.sqrt(np.sum((points[i] - points[i - 1]) ** 2))
        t.append(t[-1] + np.sqrt(chord_length))
    t = np.array(t)

    # Interpolate for x and y
    cs_x = CubicSpline(t, points[:, 0], bc_type='natural')
    cs_y = CubicSpline(t, points[:, 1], bc_type='natural')

    # Generate finely spaced t values
    t_fine = np.linspace(t[0], t[-1], num_interpolated_points)

    # Compute interpolated points
    x_interpolated = cs_x(t_fine)
    y_interpolated = cs_y(t_fine)


    """Calculate distance between each spline point"""
    from math import hypot

    def distance(p1, p2):
        """Euclidean distance between two points."""
        x1, y1 = p1
        x2, y2 = p2
        return hypot(x2 - x1, y2 - y1)

    new_coords = [(float(x[0]),float(x[1])) for x in zip(x_interpolated, y_interpolated)]
    res = [((new_coords[i]), (new_coords[i+1])) for i in range(len(new_coords)-1)]
    dists = [distance(point[0], point[1]) for point in res]
    # print(dists)

    return np.column_stack((x_interpolated, y_interpolated)), cs_x, cs_y, t_fine


def extract_rectangle_vertices(obstacles):
    """
    Given a list of rectangles, extract the x and y coordinates of non overlapped rectangles


    :param obstacles: Each rectangle is in the form [x, y, width, height] (2d list)
    :return: Two arrays, for x and y coordinates of non-overlapping rectangle vertices
    """

    def get_rectangle_vertices(rect):
        """
        Given a rectangle defined by [x, y, width, height], return the coordinates of its four vertices

        :param rect: Rectangle in the form [x, y, width, height], where (x, y) is the bottom-left corner (list)
        :return: list of tuples containing the coordinates of the four vertices
        """
        x, y, width, height = rect

        # Calculate vertices
        bottom_left = (x, y)
        bottom_right = (x + width, y)
        top_right = (x + width, y + height)
        top_left = (x, y + height)

        return [bottom_left, bottom_right, top_right, top_left]

    def is_point_inside_rectangle(point, rect):
        """
        Check if a point is inside a given rectangle.

        :param point: (x,y) point (tuple)
        :param rect: Rectangle in the form [x, y, width, height], where (x, y) is the bottom-left corner (list)
        :return: boolean
        """
        x, y = point
        rect_x, rect_y, rect_width, rect_height = rect
        return rect_x <= x <= rect_x + rect_width and rect_y <= y <= rect_y + rect_height

    all_x = []
    all_y = []

    for i, rect in enumerate(obstacles):
        vertices = get_rectangle_vertices(rect) # Get the vertices of the current rectangle

        # Check each vertex to see if it overlaps with any other rectangle
        for vertex in vertices:
            is_overlapped = any(
                is_point_inside_rectangle(vertex, other_rect)
                for j, other_rect in enumerate(obstacles)
                if i != j  # Skip the current rectangle
            )

            if not is_overlapped:
                # Append non-overlapping vertices
                all_x.append(vertex[0])
                all_y.append(vertex[1])

    return np.array(all_x), np.array(all_y)


def plot_environment(rrt, avoid_points, obstacles, path=None, optimize=False):
    """
    Plot the environment with obstacles, RRT tree, and the path
    :param rrt: rrt object
    :param avoid_points: vertices of the non-overlapping obstacles
    :param obstacles: list of obstacles [x,y,width,height]
    :param path: 2d ndarray of the path from the start goal to the end goal
    :return: None
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot obstacles
    for obs in rrt.obstacles:
        ax.add_patch(Rectangle((obs[0], obs[1]), obs[2], obs[3], color="gray"))

    # Plot the RRT tree
    tree_points = np.array(rrt.tree)
    plt.plot(tree_points[:, 0], tree_points[:, 1], 'bo', markersize=2)

    # Plot the path
    if path is not None:
        if path.ndim == 1:
            path = path.reshape(-1, 2)
        plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2)

    # TODO: Calling path_optimizer probably could go somewhere else to make it more accessible
    #       But it's easiest to put it next to all the plotting code since it can get plotted quickly too

    # Path optimization begins here
    if optimize:
        result = path_optimizer(path[:,0], path[:,1], avoid_points, obstacles, limit=12)
        optimized_spline, cs_x, cs_y, t_fine = generate_cubic_spline(result)
        plt.plot(optimized_spline[:, 0], optimized_spline[:, 1], label="Optimized Spline", color="blue")

        x, y, running_dist, curvature, v = data_restructuring(optimized_spline, cs_x, cs_y, t_fine)
        t = time_calculator(x, y, running_dist, curvature, v)

        # Create a figure and axis for the velocity plot
        fig, ax1 = plt.subplots(figsize=(8, 6))

        # Plot velocity on the primary y-axis
        ax1.plot(running_dist, v, 'b-', label="Velocity")
        ax1.set_xlabel("Running Distance")
        ax1.set_ylabel("Velocity", color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # Create a second y-axis for the curvature plot
        ax2 = ax1.twinx()
        ax2.plot(running_dist, curvature[1:], 'r-', label="Curvature")
        ax2.set_ylabel("Curvature", color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        # Add a title and legend
        fig.suptitle("Velocity and Curvature")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")

        # Show the plot
        plt.show()

        print(f"time: {t[-1]}")
        print(f"Velocity: {v}")


    # Plot obstacles
    plt.scatter(avoid_points[0], avoid_points[1], color="black", s=10)

    # Plot start and goal and other matplotlib jargon
    plt.scatter([rrt.start[0], rrt.goal[0]], [rrt.start[1], rrt.goal[1]], color='green', label='Start/Goal', s=100)
    plt.legend()
    plt.xlim(rrt.bounds[0])
    plt.ylim(rrt.bounds[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("RRT Path Planning")
    plt.grid(True)
    plt.show()