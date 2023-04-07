import numpy as np
import matplotlib.pyplot as plt
import math
import heapq

class Node:
    def __init__(self, x, y, theta, f=0, g=0, h=0, parent=None):
        self.x = x      # x-coordinate of the node
        self.y = y      # y-coordinate of the node
        self.theta = theta  # orientation angle of the node
        self.g = g      # cost-to-come from the start node
        self.h = h      # estimated cost-to-go to the goal node
        self.f = f
        self.parent = parent  # pointer to the parent node
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.theta == other.theta
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta))

class MotionModel:
    def __init__(self, steps, dt, wheelbase = 1):
        self.steps = steps
        self.wheelbase = wheelbase
        self.dt = dt

    def generate_motion_primitives(self, current_node):
        movements = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        primitives = []

        for m in movements:
            for i in range(1, self.steps+1):
                primitive = []
                for j in range(2):
                    p = m[j] * i * self.dt
                    primitive.append(p)
                primitives.append(primitive)

        trajectories = []
        for p in primitives:
            x, y, theta = current_node.x, current_node.y, current_node.theta
            for i in range(self.steps):
                theta += (p[1] / self.wheelbase) * math.tan(p[0])
                theta %= (2 * math.pi)
                x += math.cos(theta) * self.dt
                y += math.sin(theta) * self.dt
                traj = Node(x, y, theta)
                trajectories.append(traj)

        return trajectories


class HybridAStar:
    def __init__(self, occupancy_map, motion_model, resolution=0.1, goal_tolerance=0.1):
        self.occupancy_map = occupancy_map   # occupancy map or grid map
        self.open_set = []           # priority queue or heap for A* search
        self.closed_set = set()      # set of explored nodes
        self.resolution = resolution # resolution of the map
        self.goal_tolerance = goal_tolerance # tolerance for reaching the goal
        self.motion_model = motion_model # motion model object
        self.dt = motion_model.dt    # time step
        self.steps = motion_model.steps # number of steps
        self.path = []               # list of nodes in the optimal path
        self.total_cost = None       # total cost of the optimal path
        self.turning_radius = motion_model.wheelbase/ math.sin(math.radians(45))
        self.min_x = 0
        self.min_y = 0

    def euclidean_distance(self, x1, y1, x2, y2):
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    def get_trajectory(self, x, y, theta, p):
        # Convert discrete primitive to continuous trajectory
        dx = p[0] * math.cos(theta) - p[1] * math.sin(theta)
        dy = p[0] * math.sin(theta) + p[1] * math.cos(theta)
        x_traj = [x]
        y_traj = [y]
        theta_traj = [theta]
        for i in range(self.steps):
            x += dx * self.dt
            y += dy * self.dt
            theta += p[2] * self.dt
            x_traj.append(x)
            y_traj.append(y)
            theta_traj.append(theta)
        return x_traj, y_traj, theta_traj

    def check_collision(self, x, y):
        # Convert x and y to map coordinates
        map_x, map_y = self.world_to_map(x, y)

        # Check if the point is out of bounds
        if map_x < 0 or map_x >= self.occupancy_map.shape[1] or map_y < 0 or map_y >= self.occupancy_map.shape[0]:
            return True

        # Check if the point is inside an obstacle
        if self.occupancy_map[map_y, map_x] > 0.5:
            return True

        return False
    
    def world_to_map(self, x, y):
        map_x = int((x - self.min_x) / self.resolution)
        map_y = int((y - self.min_y) / self.resolution)
        return map_x, map_y
    
    def is_goal(self, node, goal):
        # Check if the node is within the goal region
        return abs(node.x - goal.x) < self.goal_tolerance and abs(node.y - goal.y) < self.goal_tolerance

    def generate_path(self, came_from, current_node):
        """
        Reconstructs the path from the start node to the current node using the came_from dictionary.
        """
        path = []
        while current_node is not None:
            # Convert the node from map coordinates to world coordinates
            x, y = self.world_to_map(current_node.x, current_node.y)
            path.append([x, y, current_node.theta])

            if current_node in came_from:
                current_node = came_from[current_node]
            else:
                break
        # Reverse the path so it goes from start to end
        path.reverse()
        return path
    
    def normalize_angle(self, angle):
        """Normalize an angle to be between -pi and pi."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def generate_successors(self, current_node, goal_node):
        successors = []
        
        # Generate motion primitives
        primitives = self.motion_model.generate_motion_primitives(current_node)
        
        for primitive in primitives:
            x = current_node.x + np.cos(current_node.theta + primitive.y) * primitive.x
            y = current_node.y + np.sin(current_node.theta + primitive.y) * primitive.x
            theta = self.normalize_angle(current_node.theta + primitive.y)
            g = current_node.g + primitive.x
            h = self.euclidean_distance(x, y, goal_node.x, goal_node.y)
            f = g + h
            
            if not self.check_collision(x, y):
                node = Node(x, y, theta, f, g, h, current_node)
                successors.append(node)
        
        return successors

    def motion_cost(self, from_node, to_node):
        """
        Calculates the cost of motion from from_node to to_node.
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        distance = np.sqrt(dx**2 + dy**2)
        angle_diff = self.normalize_angle(to_node.theta - from_node.theta)
        angle_cost = abs(angle_diff) * self.turning_radius
        return distance + angle_cost

    def search(self, start, goal, axis_plot):
        # convert start and goal to nodes
        start_x, start_y = self.world_to_map(start.x, start.y)
        goal_x, goal_y = self.world_to_map(goal.x, goal.y)
        start_node = Node(start_x, start_y, start.theta, 0, 0, 0)
        goal_node = Node(goal_x, goal_y, goal.theta, 0, 0, 0)

        # set up the open and closed lists
        open_list = []
        closed_list = set()

        # add the start node to the open list
        heapq.heappush(open_list, start_node)

        # initialize the came_from dictionary
        came_from = {}

        # continue searching until the goal is reached or the open list is empty
        while open_list:
            # get the node with the lowest cost from the open list
            current_node = heapq.heappop(open_list)

            # if the current node is the goal, return the path
            if self.is_goal(current_node, goal):
                return self.generate_path(came_from, current_node)

            # add the current node to the closed list
            closed_list.add(current_node)

            # generate the successors of the current node
            successors = self.generate_successors(current_node, goal_node)

            # iterate through the successors
            for successor in successors:
                # if the successor is already in the closed list, skip it
                if successor in closed_list:
                    continue

                # calculate the cost of the successor
                successor.g = current_node.g + self.motion_cost(current_node, successor)
                successor.h = self.euclidean_distance(successor.x, successor.y, goal_node.x, goal_node.y)
                successor.f = successor.g + successor.h

                # update the came_from dictionary
                came_from[successor] = current_node

                # if the successor is not in the open list, add it
                if successor not in open_list:
                    heapq.heappush(open_list, successor)

            path = self.generate_path(came_from, current_node)
            if path:
                path_x = [node[1] for node in path]
                path_y = [node[0] for node in path]
                axis_plot.plot(path_y, path_x, color='red')
                plt.draw()
                plt.pause(0.001)

        # if the open list is empty and the goal has not been reached, return None
        return None

# Define the main function
if __name__ == "__main__":
    # Set the start and goal nodes
    start = Node(0, 0, np.pi/4, 0, 0)
    goal = Node(20, 20, np.pi/4, 0, 0)

    # Load the occupancy map or create a random one
    occupancy_map = np.zeros((50, 50))

    # Set some obstacle locations
    obstacles = [
        (10, 10),
        (15, 20),
        (25, 30),
        (40, 10),
        (45, 45),
        (30, 20),
        (20, 5),
        (5, 35),
        (12, 45),
        (35, 15)
    ]

    # Place the obstacles in the occupancy map
    for obstacle in obstacles:
        occupancy_map[obstacle[0], obstacle[1]] = 1

    # Set the algorithm options
    motion_model = MotionModel(dt=0.1, steps=10)

    # Create a Hybrid A* object
    h = HybridAStar(occupancy_map, motion_model)

    # Visualize the results
    fig, ax = plt.subplots()
    ax.imshow(occupancy_map, cmap='gray')

    # Plot the start and goal nodes
    ax.scatter(start.y, start.x, marker='o', color='blue')
    ax.scatter(goal.y, goal.x, marker='o', color='green')

    # Run the algorithm and catch any exceptions
    h.search(start, goal, ax)

    plt.close()

