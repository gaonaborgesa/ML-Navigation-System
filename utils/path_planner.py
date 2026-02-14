"""
BFS path planning implementation.
Refactored from C++ p3 project to Python.
"""
from collections import deque
from utils.graph_utils import (
    cell_to_idx, idx_to_cell, init_graph, 
    find_neighbors, check_collision, trace_path
)


def breadth_first_search(graph, start, goal):
    """
    Perform Breadth-First Search to find path from start to goal.
    
    Args:
        graph: GridGraph object with map data
        start: Cell object representing start position
        goal: Cell object representing goal position
    
    Returns:
        List of Cell objects representing path from start to goal.
        Returns empty list if no path found.
    """
    path = []
    visit_queue = deque()
    
    # Make sure all node values are reset
    init_graph(graph)
    
    start_idx = cell_to_idx(start.i, start.j, graph)
    goal_idx = cell_to_idx(goal.i, goal.j, graph)

    # Check if start/goal are in collision    
    start_collision = check_collision(start_idx, graph)
    goal_collision = check_collision(goal_idx, graph)

    # Check if start or goal are in collision
    if start_collision:
        print("ERROR: Start position is in collision!")
        return []
    
    if goal_collision:
        print("ERROR: Goal position is in collision!")
        return []

    # Initialize start node
    graph.nodes[start_idx].cost = 0
    visit_queue.append(start_idx)
    graph.nodes[start_idx].queued = True
    

    # BFS main loop
    while visit_queue:    
        current_idx = visit_queue.popleft()
        graph.nodes[current_idx].visited = True
        
        # For visualization
        graph.visited_cells.append(idx_to_cell(current_idx, graph))
        
        # Check if we reached the goal
        if current_idx == goal_idx:
            path = trace_path(goal_idx, graph)
            print("BFS success!")
            return path
        
        # Explore neighbors
        for neighbor_idx in find_neighbors(current_idx, graph):
            neighbor_node = graph.nodes[neighbor_idx]
            
            # Only visit if not already visited/queued and not in collision
            if (not neighbor_node.visited and 
                not neighbor_node.queued and 
                not check_collision(neighbor_idx, graph)):
                
                # Set parent and cost
                neighbor_node.parent = current_idx
                neighbor_node.cost = graph.nodes[current_idx].cost + 1
                neighbor_node.queued = True
                
                # Add to queue
                visit_queue.append(neighbor_idx)
            
    # No path found
    print("No path found :(")
    return []
