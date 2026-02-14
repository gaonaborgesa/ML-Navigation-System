"""
Graph utilities for path planning on occupancy grid maps.
Refactored from C++ p3 project to Python.
"""
import math
import numpy as np


HIGH = 1e6
ROBOT_RADIUS = 0.137


class Cell:
    """Represents a cell in the grid with row (i) and column (j) indices."""
    def __init__(self, i=0, j=0):
        self.i = i
        self.j = j
    
    def __eq__(self, other):
        return self.i == other.i and self.j == other.j
    
    def __repr__(self):
        return f"Cell({self.i}, {self.j})"


class Node:
    """Node information for graph search."""
    def __init__(self):
        self.visited = False
        self.queued = False
        self.parent = -1  # Index of parent in graph data
        self.cost = HIGH  # Cost to reach cell


class GridGraph:
    """Grid-based graph representation of the map."""
    def __init__(self):
        self.width = -1
        self.height = -1
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.meters_per_cell = 0.0
        self.collision_radius = 0.15
        self.threshold = 50
        self.cell_odds = []  # Odds that each cell is occupied
        self.obstacle_distances = []  # Distance from each cell to nearest obstacle
        self.visited_cells = []  # List of visited cells for visualization
        self.nodes = []  # Node data for each cell


def load_from_file(file_path, graph):
    """
    Load map data from a file.
    
    Args:
        file_path: Path to the map file
        graph: GridGraph object to populate
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'r') as f:
            # Read header
            header = f.readline().split()
            graph.origin_x = float(header[0])
            graph.origin_y = float(header[1])
            graph.width = int(header[2])
            graph.height = int(header[3])
            graph.meters_per_cell = float(header[4])
            
            # Sanity check
            if graph.width < 0 or graph.height < 0 or graph.meters_per_cell < 0.0:
                return False
            
            # Set collision radius larger for robot
            graph.collision_radius = ROBOT_RADIUS + graph.meters_per_cell
            
            # Read cell odds
            num_cells = graph.width * graph.height
            graph.cell_odds = []
            
            # Read all remaining numbers
            for line in f:
                values = line.split()
                for val in values:
                    graph.cell_odds.append(int(val))
            
            if len(graph.cell_odds) != num_cells:
                print(f"Error: Expected {num_cells} cells, got {len(graph.cell_odds)}")
                return False
                        
            # Initialize obstacle distances
            graph.obstacle_distances = [0.0] * num_cells
            
            # Initialize nodes
            init_graph(graph)
            
            return True
            
    except Exception as e:
        print(f"Error loading map file: {e}")
        return False


def init_graph(graph):
    """Initialize the graph nodes."""
    graph.nodes = []
    num_cells = graph.width * graph.height
    for _ in range(num_cells):
        graph.nodes.append(Node())


def cell_to_idx(i, j, graph):
    """Convert cell coordinates to index in graph arrays."""
    return i + j * graph.width


def idx_to_cell(idx, graph):
    """Convert index in graph arrays to cell coordinates."""
    i = idx % graph.width
    j = idx // graph.width
    return Cell(i, j)


def pos_to_cell(x, y, graph):
    """Convert global position (meters) to cell coordinates."""
    i = int(math.floor((x - graph.origin_x) / graph.meters_per_cell))
    j = int(math.floor((y - graph.origin_y) / graph.meters_per_cell))
    return Cell(i, j)


def cell_to_pos(i, j, graph):
    """Convert cell coordinates to global position (meters)."""
    x = (i + 0.5) * graph.meters_per_cell + graph.origin_x
    y = (j + 0.5) * graph.meters_per_cell + graph.origin_y
    return [x, y]


def is_cell_in_bounds(i, j, graph):
    """Check if cell is within graph bounds."""
    return 0 <= i < graph.width and 0 <= j < graph.height


def is_idx_occupied(idx, graph):
    """Check if cell at index is occupied."""
    return graph.cell_odds[idx] >= graph.threshold


def is_cell_occupied(i, j, graph):
    """Check if cell at coordinates is occupied."""
    return is_idx_occupied(cell_to_idx(i, j, graph), graph)


def find_neighbors(idx, graph):
    """
    Find valid neighbors of a cell (8-connected grid for smoother paths).
    
    Args:
        idx: Index of the cell
        graph: GridGraph object
    
    Returns:
        List of indices of valid neighbors
    """
    neighbors = []
    current_cell = idx_to_cell(idx, graph)
    
    #check neighbors
    directions = [
        (0, -1), (0, 1), (-1, 0), (1, 0),  # Cardinal
        (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal
    ]
    
    for di, dj in directions:
        neighbor_i = current_cell.i + di
        neighbor_j = current_cell.j + dj
        
        if is_cell_in_bounds(neighbor_i, neighbor_j, graph):
            neighbors.append(cell_to_idx(neighbor_i, neighbor_j, graph))
    
    return neighbors


def check_collision(idx, graph):
    """
    Check if robot would collide at this cell.
    SIMPLIFIED VERSION - just check the cell itself for obstacle-free environment.
    
    Args:
        idx: Index of the cell
        graph: GridGraph object
    
    Returns:
        True if collision detected, False otherwise
    """
    # Values > 50 are obstacles, < 0 are free space, ~0 is unknown
    return graph.cell_odds[idx] > 50


def get_parent(idx, graph):
    """Get parent index of a node."""
    return graph.nodes[idx].parent


def trace_path(goal_idx, graph):
    """
    Trace path from goal back to start using parent pointers.
    
    Args:
        goal_idx: Index of goal cell
        graph: GridGraph object
    
    Returns:
        List of Cell objects from start to goal
    """
    path = []
    current = goal_idx
    
    while current >= 0:  # A cell with no parent has parent -1
        path.append(idx_to_cell(current, graph))
        current = get_parent(current, graph)
    
    # Reverse since we built path backwards
    path.reverse()
    return path


def cells_to_poses(path, graph):
    """
    Convert cell path to pose path (x, y, theta).
    
    Args:
        path: List of Cell objects
        graph: GridGraph object
    
    Returns:
        List of [x, y, theta] poses
    """
    pose_path = []
    
    for i, cell in enumerate(path):
        position = cell_to_pos(cell.i, cell.j, graph)
        
        # Calculate theta based on direction to next waypoint
        theta = 0.0
        if i < len(path) - 1:
            next_cell = path[i + 1]
            next_pos = cell_to_pos(next_cell.i, next_cell.j, graph)
            dx = next_pos[0] - position[0]
            dy = next_pos[1] - position[1]
            theta = math.atan2(dy, dx)
        
        pose = [position[0], position[1], theta]
        pose_path.append(pose)
    
    return pose_path