'''
EECE Project
Smooth Path Planning as an Optimization Problem
Author: Ryan Huang

The reward function using several penalties

Minimizing the following:

    Derivatives
    Curvature
    Path Length
    Distance to obstacles
    Collisions with obstacles
    Derivation from the ideal straight line path
'''
import numpy as np
from path_time import data_restructuring, time_calculator

# TODO: Refactor the reward function to turn the penalties into function inputs

def reward_function(optimized_coords, x_initial, y_fixed, avoid_set, obstacles, curvature_penalty):
    """
    Compute the negative reward for optimization, including a penalty for small radius of curvature.

    :param optimized_coords: Intermediate coordinates to optimize
    :param x_initial: Original fixed x coordinates
    :param y_fixed: Original fixed y coordinates
    :param avoid_set: Outermost vertices of the obstacles
    :param obstacles: list of obstacles [x,y,width,height]
    :return: Reward (float)
    """

    # Split the optimized coordinates back into x and y
    intermediate_x = optimized_coords[:len(x_initial) - 2]
    intermediate_y = optimized_coords[len(x_initial) - 2:]

    # Reconstruct full x-coordinates (start + intermediate + end)
    x_coords = np.concatenate(([x_initial[0]], intermediate_x, [x_initial[-1]]))
    y_coords = np.concatenate(([y_fixed[0]], intermediate_y, [y_fixed[-1]]))

    '''
    OBSTACLE DISTANCE PENALTY
    ------------------------------------------------------------------------------------------
    '''

    min_distance = 5
    avoid_x, avoid_y = avoid_set[0], avoid_set[1]     # Extract unwanted vertices
    collision_penalty = 0


    for ax, ay in zip(avoid_x, avoid_y):
        # Compute distances from this avoid point to all spline points
        distances = np.sqrt((intermediate_x - ax) ** 2 + (intermediate_y - ay) ** 2)

        # Check if any spline point is within the threshold distance
        if np.min(distances) < min_distance:
            collision_penalty += 0

    '''
    Needed some extra functions and assertions
    ------------------------------------------------------------------------------------------
    '''
    # Ensure the dimensions match (length of x_coords and y_coords should match)
    assert len(x_coords) == len(y_coords), f"Mismatch in lengths: {len(x_coords)} vs {len(y_coords)}"

    from utility import calculate_spline_length, generate_cubic_spline

    '''
    DERIVATIVE PENALTY
    ------------------------------------------------------------------------------------------
    '''
    # Generate the spline
    points = np.column_stack((x_coords, y_coords))
    spline_points, cs_x, cs_y, t_fine = generate_cubic_spline(points)

    # Compute first and second derivatives
    dx_dt = cs_x.derivative()(t_fine)
    dy_dt = cs_y.derivative()(t_fine)
    d2x_dt2 = cs_x.derivative(2)(t_fine)
    d2y_dt2 = cs_y.derivative(2)(t_fine)

    # Compute tangent and curvature norms
    norm_tangent = np.sqrt(dx_dt**2 + dy_dt**2)
    norm_curvature = np.sqrt(d2x_dt2**2 + d2y_dt2**2)


    '''
    COLLISION PENALTY
    ------------------------------------------------------------------------------------------
    '''

    for obst in obstacles:
        x, y, width, height = obst
        min_x, max_x = x, x + width
        min_y, max_y = y, y + height

        for sp_x, sp_y in spline_points:
            if min_x <= sp_x <= max_x and min_y <= sp_y <= max_y:
                collision_penalty += 100000

    '''
    CURVATURE PENALTY
    ------------------------------------------------------------------------------------------
    '''
    # Compute the radius of curvature at each point
    numerator = (dx_dt**2 + dy_dt**2)**(3/2)
    denominator = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)  # Curvature formula for radius
    radius_of_curvature = numerator / denominator

    # Combine rewards: penalize small radius (sharp turns) and large curvature
    curvature_penalty_term = curvature_penalty * np.sum(1 / radius_of_curvature)  # Penalize small radii

    '''
    STRAIGHT LINE PENALTY 
    ------------------------------------------------------------------------------------------
    '''
    #TODO: Unused, this is a little strong. Maybe come up with a discount factor

    # Define the straight line between the first and last points
    x_start, y_start = x_initial[0], y_fixed[0]
    x_end, y_end = x_initial[-1], y_fixed[-1]

    # Calculate slope and then the y-coordinates using the slope
    m = (y_end - y_start) / (x_end - x_start)  # slope
    straight_line_y = m * (x_coords - x_start) + y_start  # y-values

    # Interpolate the line using the same resolution as the spline
    straight_line_y_interp = np.interp(spline_points[:, 0], x_coords, straight_line_y)

    # Calculate the distance from the spline and the straight line
    deviation = np.abs(spline_points[:, 1] - straight_line_y_interp)  # Deviation in the y-direction

    # Penalize the deviation from the straight line
    straight_line_penalty = np.sum(deviation**2)  # Sum of squared deviations

    '''
    LENGTH PENALTY
    ------------------------------------------------------------------------------------------
    '''

    # Compute total length of the spline
    total_length = calculate_spline_length(cs_x, cs_y, t_fine[0], t_fine[-1])


    '''
    TIME PENALTY
    ------------------------------------------------------------------------------------------
    '''

    x, y, running_dist, curvature, v  = data_restructuring(spline_points, cs_x, cs_y, t_fine)
    time_array = time_calculator(x, y, running_dist, curvature, v)
    time_penalty = 100000*time_array[-1]



    # Final reward function with length, curvature, and radius of curvature penalties
    # reward = np.sum(norm_tangent) + np.sum(norm_curvature) + curvature_penalty_term + total_length + straight_line_penalty
    # reward = -np.sum(norm_tangent) - np.sum(norm_curvature) + curvature_penalty_term + 150*total_length + collision_penalty + time_penalty

    reward = time_penalty + 150*total_length + curvature_penalty_term + collision_penalty
    # print(f"Reward: {reward}")
    # print(f"1st Derivative: {-np.sum(norm_tangent):.8f} 2nd Derivative: {-np.sum(norm_curvature):.8f} Curvature: {curvature_penalty_term:.8f} Total Length: {total_length:.8f} Collision {collision_penalty} Time: {time_penalty}")
    # print(f"Reward: {reward}")
    print(f"Time: {time_penalty} Total Length: {total_length:.8f} Curvature: {curvature_penalty_term:.8f}  Collision {collision_penalty}")
    return reward  # Negate for minimization