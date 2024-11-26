'''
EECE Project
Smooth Path Planning as an Optimization Problem
Author: Ryan Huang

Path optimizer via scipy.optimize.minimize
'''
from utility import *
from scipy.optimize import minimize
from reward import reward_function


def path_optimizer(x_initial, y_fixed, avoid_set, limit):
    """
    Optimize the path using scipy.optimize.minimize

    :param x_initial: Original fixed x coordinates (ndarray)
    :param y_fixed: Original fixed y coordinates (ndarray)
    :param avoid_set: Outermost vertices of the obstacles (ndarray)
    :param limit: The rectangular bound on each spline point to optimize (float)
    :return: Optimized points (ndarray)
    """
    # Exclude the start and goal points from optimization
    # These points cannot shift or it defeats the point of path optimization
    intermediate_x = x_initial[1:-1]
    intermediate_y = y_fixed[1:-1]
    initial_guess = np.concatenate([intermediate_x, intermediate_y])

    # The optimization aspect comes from allowing the intermediate points to shift
    # These lines impose bounds on how far they can shift
    bounds_x = [(x - limit, x + limit) for x in x_initial[1:-1]]
    bounds_y = [(y - limit, y + limit) for y in y_fixed[1:-1]]

    bounds = bounds_x + bounds_y

    curvature_penalty = 200  # Strong penalty for sharp turns

    # Using scipy.minimize and the SLSQP method to minimize the reward function
    result = minimize(
        reward_function,
        initial_guess,
        args=(x_initial, y_fixed, avoid_set, curvature_penalty),
        method='SLSQP',
        bounds=bounds,
        options={'disp': True},
    )

    # Extract the optimized coordinates
    optimized_coords = result.x
    optimized_x = np.concatenate(([x_initial[0]], optimized_coords[:len(x_initial)-2], [x_initial[-1]]))
    optimized_y = np.concatenate(([y_fixed[0]], optimized_coords[len(x_initial)-2:], [y_fixed[-1]]))

    # Reformatting points for post processing
    optimized_points = np.column_stack((optimized_x, optimized_y))

    return optimized_points



if __name__ == "__main__":
    '''
    Run this file to test point generation and path optimizing
    
    It's essentially a very miniature RRT and spline generator
    Also tests the path optimizer function
    '''

    original_points = generate_vertical_points(num_points=8, vertical_spread=50, horizontal_spread=20)
    x_initial = original_points[:, 0]
    y_fixed = original_points[:, 1]

    optimized_points = path_optimizer(x_initial, y_fixed, [[1,1], [2,2]], limit=50)

    # Generate splines for visualization
    original_spline, _, _, _ = generate_cubic_spline(original_points)
    optimized_spline, _, _, _ = generate_cubic_spline(optimized_points)


    # Miscellaneous matplotlib jargon to make the graphs look pretty
    plt.figure(figsize=(8, 12))
    plt.plot(original_points[:, 0], original_points[:, 1], marker='o', linestyle='-', color='black', label='Connected Points')
    plt.plot(original_spline[:, 0], original_spline[:, 1], label="Original Spline", color="blue")
    plt.plot(optimized_spline[:, 0], optimized_spline[:, 1], label="Optimized Spline", color="green")
    plt.scatter(original_points[:, 0], original_points[:, 1], color="red", label="Original Points")
    plt.scatter(optimized_points[:, 0], optimized_points[:, 1], color="purple", label="Optimized Points")
    plt.title("Original vs Optimized Spline")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.show()


