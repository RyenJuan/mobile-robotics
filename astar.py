import heapq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class AStar:
    def __init__(self, start, goal, bounds, obstacles):
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.obstacles = obstacles
        self.open_list = []
        self.closed_list = set()
        self.came_from = {}
        self.g_score = {start: 0}
        self.f_score = {start: self.heuristic(start)}

    def heuristic(self, node):
        """Euclidean distance heuristic for diagonal movement."""
        return np.sqrt((node[0] - self.goal[0]) ** 2 + (node[1] - self.goal[1]) ** 2)

    def is_valid(self, node):
        """Check if the node is within bounds and not colliding with obstacles."""
        if node[0] < self.bounds[0][0] or node[0] > self.bounds[0][1]:
            return False
        if node[1] < self.bounds[1][0] or node[1] > self.bounds[1][1]:
            return False

        for obs in self.obstacles:
            obs_x, obs_y, obs_w, obs_h = obs
            # Check if the node is inside the obstacle or on its boundary
            if (obs_x <= node[0] <= obs_x + obs_w) and (obs_y <= node[1] <= obs_y + obs_h):
                return False  # Node is inside or on the boundary of the obstacle

        return True

    def get_neighbors(self, node):
        """Get valid neighboring nodes (8-connected neighbors)."""
        neighbors = [
            (node[0] + 1, node[1]),  # Right
            (node[0] - 1, node[1]),  # Left
            (node[0], node[1] + 1),  # Up
            (node[0], node[1] - 1),  # Down
            (node[0] + 1, node[1] + 1),  # Top-right diagonal
            (node[0] - 1, node[1] + 1),  # Top-left diagonal
            (node[0] + 1, node[1] - 1),  # Bottom-right diagonal
            (node[0] - 1, node[1] - 1),  # Bottom-left diagonal
        ]
        return [neighbor for neighbor in neighbors if self.is_valid(neighbor)]

    def reconstruct_path(self, current):
        """Reconstruct the path from start to goal."""
        path = []
        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start)
        return np.array(path[::-1])

    def search(self):
        """Perform the A* search algorithm."""
        heapq.heappush(self.open_list, (self.f_score[self.start], self.start))

        while self.open_list:
            _, current = heapq.heappop(self.open_list)

            # If we reached the goal, reconstruct the path
            if current == self.goal:
                return self.reconstruct_path(current)

            self.closed_list.add(current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = self.g_score.get(current, float('inf')) + (
                    np.sqrt(2) if abs(neighbor[0] - current[0]) + abs(neighbor[1] - current[1]) == 2 else 1)

                if neighbor in self.closed_list:
                    continue

                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[neighbor] = self.g_score[neighbor] + self.heuristic(neighbor)
                    heapq.heappush(self.open_list, (self.f_score[neighbor], neighbor))

        return None  # No path found


def plot_environment(start, goal, obstacles, path=None):
    """Plot the environment, obstacles, and the found path."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot obstacles
    for obs in obstacles:
        rect = patches.Rectangle((obs[0], obs[1]), obs[2], obs[3], linewidth=1, edgecolor='r', facecolor='r', alpha=0.5)
        ax.add_patch(rect)

    # Plot the start and goal points
    ax.plot(start[0], start[1], 'go', label='Start')
    ax.plot(goal[0], goal[1], 'bo', label='Goal')

    # Plot the path if it exists
    if path is not None:
        ax.plot(path[:, 0], path[:, 1], 'g-', linewidth=2, label='Path')

    # Set axis limits and labels
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 60)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('A* Pathfinding')
    ax.legend()

    # Show the plot
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Define environment
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

    # Create an A* object and find the path
    a_star = AStar(start, goal, bounds, obstacles)
    path = a_star.search()

    # Output the result
    if path is not None:
        print("Path found:")
        print(path)
        plot_environment(start, goal, obstacles, path)
    else:
        print("No path found.")
