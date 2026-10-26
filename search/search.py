# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# # Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in search_agents.py).
"""
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in obj-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem.
        """
        util.raise_not_defined()

    def is_goal_state(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raise_not_defined()

    def get_successors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raise_not_defined()

    def get_cost_of_actions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raise_not_defined()


def tiny_maze_search(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

# def addSuccessors(problem, addCost=True):

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.parent = parent

    # The coordinates of a node cannot be modified, se we just define a getter.
    # This allows the class to be hashable.
    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
    
    # Consider 2 nodes to be equal if their coordinates are equal (regardless of everything else)
    # def __eq__(self, __o: obj) -> bool:
    #     if (type(__o) is SearchNode):
    #         return self.__state == __o.__state
    #     return False

    # # def __hash__(self) -> int:
    # #     return hash(self.__state)

def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state())
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """

    expanded_nodes = set()
    frontier = util.Stack()

    # Initialize frontier
    initial_node = problem.get_start_state()
    frontier.push(initial_node)

    # Dictionary to store paths to each node
    moves_to_node = {}
    moves_to_node[initial_node] = []

    while not frontier.is_empty():
        # Choose a node from frontier
        node = frontier.pop()
        expanded_nodes.add(node)
        #print(f"CHOSE {node} FROM FRONTIER")

        if problem.is_goal_state(node):
            return moves_to_node[node]
        
        #print(f"{node} ISN'T GOAL STATE")

        for next_state in problem.get_successors(node):
            (childNode, action, cost) = next_state

            if childNode not in expanded_nodes:
                #print(f"{childNode} NOT VISITED YET ")

                frontier.push(childNode)
                path_to_parent = moves_to_node[node]
                moves_to_node[childNode] = path_to_parent + [action]
                #print(f"START TO PARENT {node}: {path_to_parent}")
                #print(f"FULL PATH TO {childNode}: {moves_to_node[childNode]}")

    return []



def breadth_first_search(problem):
    """Search the shallowest nodes in the search tree first."""
    expanded_nodes = set()
    frontier = util.Queue()

    moves_to_node = {}

    initial_node = problem.get_start_state()
    frontier.push(initial_node)
    moves_to_node[initial_node] = []
    #print(f"INITIAL STATE: {initial_node}")

    while not frontier.is_empty():
        # Choose a node from frontier
        node = frontier.pop()
        expanded_nodes.add(node)
        #print(f"CHOSE {node} FROM FRONTIER")

        if problem.is_goal_state(node):
            return moves_to_node[node]
        
        #print(f"{node} ISN'T GOAL STATE")

        for next_state in problem.get_successors(node):
            (childNode, action, cost) = next_state

            if childNode not in expanded_nodes and childNode not in frontier.list:
                #print(f"{childNode} NOT VISITED YET ")

                # Add child to frontier at the end of the queue
                frontier.push(childNode)

                path_to_parent = moves_to_node[node]
                moves_to_node[childNode] = path_to_parent + [action]
                #print(f"START TO PARENT {node}: {path_to_parent}")
                #print(f"FULL PATH TO {childNode}: {moves_to_node[childNode]}")

    return []

def uniform_cost_search(problem):
    """Search the node of least total cost first."""
    expanded_nodes = set()
    frontier = util.PriorityQueue()

    initial_node = problem.get_start_state()
    # Push a tuple of (state, path, cost) and use cost as the priority
    frontier.push((initial_node, [], 0), priority=0)

    while not frontier.is_empty():
        # Choose a node from frontier
        node, path, current_cost = frontier.pop()

        # If we have already expanded this node (e.g. from a duplicate push), skip it
        if node in expanded_nodes:
            continue
            
        expanded_nodes.add(node)

        if problem.is_goal_state(node):
            return path

        for next_state in problem.get_successors(node):
            childNode, action, stepCost = next_state

            if childNode not in expanded_nodes:
                new_cost = current_cost + stepCost
                new_path = path + [action]
                
                # Push every valid child. The PriorityQueue will naturally sort them.
                # If duplicates are pushed, the one with the lowest cost is popped first, 
                # and the more expensive duplicates will be caught by the `node in expanded_nodes` check later.
                frontier.push((childNode, new_path, new_cost), priority=new_cost)

    return []

def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def a_star_search(problem, heuristic=null_heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    expanded_nodes = set()
    frontier = util.PriorityQueue()

    initial_node = problem.get_start_state()
    # Push a tuple of (state, path, cost) and use cost as the priority
    frontier.push((initial_node, [], 0), priority=0)

    while not frontier.is_empty():
        # Choose a node from frontier
        node, path, current_cost = frontier.pop()

        # If we have already expanded this node (e.g. from a duplicate push), skip it
        if node in expanded_nodes:
            continue
            
        expanded_nodes.add(node)

        if problem.is_goal_state(node):
            return path

        for next_state in problem.get_successors(node):
            childNode, action, stepCost = next_state

            if childNode not in expanded_nodes:
                new_cost = current_cost + stepCost
                priority = new_cost + heuristic(childNode, problem)
                new_path = path + [action]
                
                # Push every valid child. The PriorityQueue will naturally sort them.
                # If duplicates are pushed, the one with the lowest cost is popped first, 
                # and the more expensive duplicates will be caught by the `node in expanded_nodes` check later.
                frontier.push((childNode, new_path, new_cost), priority=priority)

    return []

# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
