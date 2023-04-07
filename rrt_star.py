import random
import math
import numpy as np
import matplotlib.pyplot as plt

OBSTACLE_SIZE = 0.1
GAP_SIZE = 0.1

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacles, x_min, x_max, y_min, y_max, max_iter, step_size, near_radius, goal_radius, goal_sample_rate):
        self.start_node = start
        self.goal_node = goal
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.max_iter = max_iter
        self.step_size = step_size
        self.near_radius = near_radius
        self.goal_radius = goal_radius
        self.goal_sample_rate = goal_sample_rate
        self.obstacle_list = obstacles
        self.node_list = [self.start_node]

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = [random.uniform(self.x_min, self.x_max),
                random.uniform(self.y_min, self.y_max)]
        else:  # goal point sampling
            rnd = [self.goal_node.x, self.goal_node.y]
        node = Node(rnd[0], rnd[1])
        return node

    def get_nearest_node_index(self, random_node):
        """
        Returns the index of the node in the tree closest to the randomly generated node.
        """
        distance_list = [(node.x - random_node.x) ** 2 + (node.y - random_node.y) ** 2 for node in self.node_list]
        min_index = distance_list.index(min(distance_list))
        return min_index
    
    def calc_distance_and_angle(self, start, end):
        dx = end.x - start.x
        dy = end.y - start.y
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        return distance, angle
    
    def calc_dist(self, start, end):
        dx = end.x - start.x
        dy = end.y - start.y
        distance = math.sqrt(dx**2 + dy**2)
        return distance
    
    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        dist, angle = self.calc_distance_and_angle(new_node, to_node)
        if extend_length > dist:
            extend_length = dist
        new_node.cost = from_node.cost + extend_length
        new_node.x += extend_length * math.cos(angle)
        new_node.y += extend_length * math.sin(angle)
        new_node.parent = self.node_list.index(from_node)
        return new_node

    def point_inside_obstacle(self, point, obs):
        dist = math.sqrt((point.x - obs.x)**2 + (point.y - obs.y)**2)
        if dist <= OBSTACLE_SIZE + GAP_SIZE:
            return True
        return False

    def line_intersect_obstacle(self, start, end, obstacle):
        # calculate the distance between the obstacle center and the line segment
        dist = self.point_line_distance([obstacle.x, obstacle.y], [start.x, start.y], [end.x, end.y])
        
        # if the distance is less than or equal to the obstacle radius, there is an intersection
        if dist <= OBSTACLE_SIZE + GAP_SIZE:
            return True
                
        # no intersections found
        return False

    def point_line_distance(self, point, line_start, line_end):
        """Calculate the distance between a point and a line segment defined by two points."""

        # Calculate the vector between the line start and end points
        line_vector = np.array(line_end) - np.array(line_start)
        
        # Calculate the vector between the line start point and the point
        point_vector = np.array(point) - np.array(line_start)
        
        # Calculate the length of the line segment
        line_length = np.linalg.norm(line_vector)
        
        # Calculate the dot product of the line vector and the point vector
        dot_product = np.dot(line_vector, point_vector)
        
        # Calculate the projection of the point vector onto the line vector
        projection = dot_product / line_length ** 2
        
        # Calculate the closest point on the line segment to the point
        closest_point = np.array(line_start) + projection * line_vector
        
        # Calculate the distance between the point and the closest point on the line segment
        distance = np.linalg.norm(np.array(point) - closest_point)
        
        return distance

    def check_collision(self, node1, node2):
        """
        Checks if there is a collision between two nodes.
        """
        for obstacle in self.obstacle_list:
            if self.line_intersect_obstacle(node1, node2, obstacle) or self.point_inside_obstacle(node1, obstacle) or self.point_inside_obstacle(node2, obstacle):
                return False
        return True

    def find_near_nodes(self, new_node):
        n = len(self.node_list) + 1
        r = self.near_radius * math.sqrt((math.log(n) / n))
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        near_inds = [ind for ind, d in enumerate(dist_list) if d <= r**2]
        return near_inds

    def choose_parent(self, near_inds, new_node):
        if not near_inds:
            return None
        
        # Compute the costs of reaching the nearby nodes from the new node
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            d = self.calc_dist(near_node, new_node)
            if self.check_collision(near_node, new_node):
                costs.append(near_node.cost + d)
            else:
                costs.append(float('inf'))

        # Choose the nearby node that results in the lowest-cost path
        min_cost = min(costs)
        min_ind = near_inds[costs.index(min_cost)]
        if min_cost == float('inf'):
            return None  # All nearby nodes are in collision

        # Create a new node with the lowest-cost path
        new_node.cost = min_cost
        new_node.parent = min_ind

        return min_ind


    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            if self.check_collision(new_node, near_node):
                new_cost = near_node.cost + self.calc_dist(new_node, near_node)
                if new_cost < new_node.cost:
                    new_node.parent = self.node_list.index(near_node)
                    new_node.cost = new_cost

    def find_path(self):
        # if self.node_list[-1] != self.goal_node:
        #     return None

        path = []
        node = self.node_list[-1]
        while node.parent is not None:
            path.append(node)
            node = self.node_list[node.parent]

        path.append(self.start_node)
        return list(reversed(path))

    def planning(self, axis_plot):
        self.node_list = [self.start_node]

        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.step_size)
            if self.check_collision(new_node, nearest_node):
                near_inds = self.find_near_nodes(new_node)
                edge_node = self.choose_parent(near_inds, new_node)
                if edge_node is not None:
                    self.rewire(new_node, near_inds)
                    self.node_list.append(new_node)

            # path = self.find_path()
            # if path is not None:
            #     x_vals = [node.x for node in path]
            #     y_vals = [node.y for node in path]
            #     axis_plot.plot(x_vals, y_vals, 'b-')
            #     plt.draw()
            #     plt.pause(0.001)

            if self.calc_dist(self.node_list[-1], self.goal_node) <= self.goal_radius:
                final_node = self.steer(self.node_list[-1], self.goal_node, self.step_size)
                if self.check_collision(final_node, self.node_list[-1]):
                    self.node_list.append(final_node)
                    path = self.find_path()
                    return path

        return None

if __name__ == "__main__":
    # Define the problem boundaries
    x_min, x_max = 0, 10
    y_min, y_max = 0, 10

    # Define the start and goal points
    start = Node(x=1, y=1)
    goal = Node(x=9, y=9)

    # Define a list of obstacles
    obstacles = [Node(3, 3), Node(4, 3), Node(5, 3), Node(6, 3), Node(7, 3),
                Node(3, 7), Node(4, 7), Node(5, 7), Node(6, 7), Node(7, 7)]

    # Define the RRT* algorithm parameters
    max_iter = 10000
    step_size = 0.1
    near_radius = 1
    goal_radius = 0.2
    goal_sample_rate = 30

    # Create an RRT* planner instance
    rrt_star = RRTStar(start, goal, obstacles, x_min, x_max, y_min, y_max, max_iter, step_size, near_radius, goal_radius, goal_sample_rate)

    # Plot the result
    fig, ax = plt.subplots()
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    # Plot the obstacles
    for obstacle in obstacles:
        circle = plt.Circle((obstacle.x, obstacle.y), OBSTACLE_SIZE, color='gray')
        ax.add_artist(circle)

    # Plot the start and goal points
    ax.plot(start.x, start.y, 'go', markersize=10)
    ax.plot(goal.x, goal.y, 'ro', markersize=10)

    # Plan a path using the RRT* algorithm
    path = rrt_star.planning(ax)

    # Plot the planned path
    if path is not None:
        x_vals = [node.x for node in path]
        y_vals = [node.y for node in path]
        ax.plot(x_vals, y_vals, 'b-')
        plt.show()
    else:
        print("Failed to find a path!")
