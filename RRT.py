'''
EECE Project
Smooth Path Planning as an Optimization Problem
Author: Ryan Huang

Rapid Random Tree class implementation
'''

from utility import *

class RRT:
    """
    RRT Implementation. Creates the environment, obstacles, start, goal, and solves for the path
    """
    def __init__(self, start, goal, bounds, obstacles, step_size=1, max_iter=2000):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.tree = [self.start]
        self.parent_map = {}

    def generate_random_point(self):
        """
        Generate a random (x,y) point in the bounded space
        :return: Random point (nd array)
        """
        x = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
        y = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
        return np.array([x, y])

    def distance(self, point1, point2):
        """
        Calculate the euclidian distance between two (x,y) points
        :param point1: ndarray
        :param point2: ndarray
        :return: Euclidian distance (float
        """
        return np.linalg.norm(point1 - point2)

    def nearest_neighbor(self, point):
        """
        Find the nearest node in the tree
        :param point: (x,y) (ndarray)
        :return:
        """
        distances = [self.distance(point, node) for node in self.tree]
        nearest_index = np.argmin(distances)
        return self.tree[nearest_index]

    def check_collision(self, start, end):
        """
        Check if the line segment from 'start' to 'end' intersects any obstacles
        :param start: ndarray
        :param end: ndarray
        :return: boolean
        """
        for obs in self.obstacles:
            if self.line_intersects_rectangle(start, end, obs):
                return True
        return False

    def line_intersects_rectangle(self, start, end, rect):
        """
        Check if a line segment intersects a rectangle (obstacle)
        :param start: line segment start (ndarray)
        :param end: line segment end (ndarray)
        :param rect: obstacle left corner coordinates, width, and height (ndarray)
        :return: boolean
        """
        # Increase the detection zone to be conservative
        padding = 3
        rect_x, rect_y, rect_width, rect_height = rect
        # Check if the line intersects with any of the four sides of the rectangle
        # Using line parameterization to detect intersection
        dx, dy = end - start
        t1 = (rect_x - padding - start[0]) / dx if dx != 0 else (rect_y - padding - start[1]) / dy
        t2 = (rect_x + rect_width + padding - start[0]) / dx if dx != 0 else (rect_y + rect_height + padding - start[1]) / dy
        t3 = (rect_y - padding - start[1]) / dy if dy != 0 else (rect_x - padding - start[0]) / dx
        t4 = (rect_y + rect_height + padding - start[1]) / dy if dy != 0 else (rect_x + rect_width + padding - start[0]) / dx

        tmin = max(min(t1, t2), min(t3, t4))
        tmax = min(max(t1, t2), max(t3, t4))

        if tmax < 0 or tmin > 1:
            return False
        return True

    def plan(self):
        """
        Plan the path from start to goal using RRT
        :return: None
        """
        for i in range(self.max_iter):
            rand_point = self.generate_random_point()
            nearest = self.nearest_neighbor(rand_point)
            direction = (rand_point - nearest) / self.distance(rand_point, nearest)
            new_point = nearest + direction * self.step_size

            # Check for collisions
            if not self.check_collision(nearest, new_point):
                self.tree.append(new_point)
                self.parent_map[tuple(np.round(new_point, decimals=6))] = nearest  # Ensure rounding

                # If goal is reached
                if self.distance(new_point, self.goal) < self.step_size:
                    self.tree.append(self.goal)
                    self.parent_map[tuple(np.round(self.goal, decimals=6))] = new_point  # Ensure rounding
                    break

    def get_path(self):
        """
        Retrieve the path from start to goal
        :return: Path as a 2D ndarray
        """
        # FIXME: Depending on random luck, occasionally the goal itself is not appended to the path causing a key error
        #        Since the error only occurs randomly, I'm willing to overlook it for now
        #Traceback (most recent call last):
        #     path = rrt.get_path()
        #            ^^^^^^^^^^^^^^
        #     current_node = self.parent_map[tuple(np.round(current_node, decimals=6))]
        #                    ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # KeyError: (np.int64(50), np.int64(50))
        #

        if tuple(np.round(self.goal, decimals=6)) not in self.parent_map:
            raise KeyError("Goal is not connected to the tree!")

        path = [self.goal]
        current_node = self.goal
        while not np.array_equal(current_node, self.start):
            current_node = self.parent_map[tuple(np.round(current_node, decimals=6))]
            path.append(current_node)
        return np.array(path[::-1])

if __name__ == "__main__":
    # Define the basic environment
    start = (0, 0)
    goal = (50, 50)
    bounds = [(0, 60), (0, 60)] # Extra padding to aid visualization
    num_obstacles = 5
    obstacles = []

    # Random Obstacle Generator
    for i in range(num_obstacles):
        location = np.random.randint(10, 40, (1, 2))
        size = np.random.randint(5, 20, (1, 2))
        obst = np.concatenate((location, size)).reshape(1,4)
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
    cut_path = path[0::2]
    if goal not in cut_path:
        cut_path = np.append(cut_path, goal)

    # Plot the environment and the resulting path
    plot_environment(rrt, [avoid_x, avoid_y], obstacles, path=cut_path, optimize=False)
