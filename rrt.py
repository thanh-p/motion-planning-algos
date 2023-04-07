import math
import random
import matplotlib.pyplot as plt

OBSTACLE_SIZE = 0.1
CAR_SIZE = 0.1
GAP_SIZE = 0.1
GOAL_SIZE = 0.2

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class Tree:
    def __init__(self, start):
        self.nodes = [start]

def RRT(start, goal, obstacles, num_iterations):
    tree = Tree(start)
    counter = 0
    for i in range(num_iterations):
        q_rand = Node(random.uniform(0, 10), random.uniform(0, 10)) # Randomly sample a configuration in the space
        q_near = nearest_neighbor(q_rand, tree) # Find the nearest node in the tree
        q_new = extend(q_near, q_rand, obstacles) # Extend the tree from the nearest node towards the new node
        if q_new is not None:
            tree.nodes.append(q_new) # Add the new node to the tree
            if is_goal_reachable(q_new, goal, obstacles): # Check if the goal is reachable from the new node
                visualize_tree(tree, obstacles, goal, construct_path(q_new, tree), True)
                return construct_path(q_new, tree) # If yes, construct the path and return it
        
        counter += 1
        if counter %10 == 0:
            visualize_tree(tree, obstacles, goal, construct_path(q_new, tree))
    return None # If the goal is not reachable within the maximum number of iterations, return None

def nearest_neighbor(q, tree):
    min_dist = float('inf')
    nearest_node = None
    for node in tree.nodes:
        dist = euclidean_distance(q, node)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    return nearest_node

def extend(q_near, q_rand, obstacles):
    step_size = 0.1
    theta = math.atan2(q_rand.y - q_near.y, q_rand.x - q_near.x)
    q_new = Node(q_near.x + step_size * math.cos(theta), q_near.y + step_size * math.sin(theta))
    if not is_collision(q_new, obstacles):
        q_new.parent = q_near
        return q_new
    return None

def is_collision(q, obstacles):
    for obstacle in obstacles:
        if euclidean_distance(q, obstacle) < OBSTACLE_SIZE + CAR_SIZE + GAP_SIZE:
            return True
    return False

def is_goal_reachable(q, goal, obstacles):
    return not is_collision(q, obstacles) and euclidean_distance(q, goal) < GOAL_SIZE

def euclidean_distance(p1, p2):
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.sqrt(dx**2 + dy**2)

def construct_path(q_new, tree):
    path = []
    current_node = q_new
    while current_node is not None:
        path.append(current_node)
        current_node = current_node.parent
    path.append(tree.nodes[0])
    path.reverse()
    return path

def visualize_tree(tree, obstacles, goal, path=None, pause=False):
    plt.clf()
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    for node in tree.nodes:
        if node.parent is not None:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], 'k-')
    for obstacle in obstacles:
        circle = plt.Circle((obstacle.x, obstacle.y), OBSTACLE_SIZE, color='r')
        plt.gcf().gca().add_artist(circle)

    circle = plt.Circle((goal.x, goal.y), GOAL_SIZE, color='b')
    plt.gcf().gca().add_artist(circle)
    
    if path is not None:
        plt.plot([node.x for node in path], [node.y for node in path], 'r-', linewidth=2)
    
    if not pause:
        plt.draw()
        plt.pause(0.001)
    else:
        plt.show()


if __name__ == "__main__":
    start = Node(0, 0)
    goal = Node(9, 9)
    obstacles = [Node(3, 3), Node(4, 3), Node(5, 3), Node(6, 3), Node(7, 3), Node(3, 7), Node(4, 7), Node(5, 7), Node(6, 7), Node(7, 7)]
    num_iterations = 5000
    path = RRT(start, goal, obstacles, num_iterations)
    if path is not None:
        print("Path found!")
        for node in path:
            print(node)
    else:
        print("Path not found.")
