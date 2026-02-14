import subprocess
import time
import math

from mbot_bridge.api import MBot
from utils.graph_utils import (
    GridGraph, load_from_file, pos_to_cell, 
    cells_to_poses, Cell
)
from utils.path_planner import breadth_first_search

MAP_PATH = "/home/mbot/current.map"
TURN_KP = 5
MAX_TURN_SPEED = 1.5
TURN_ERROR = 0.03
POSE_ERROR = 0.03
PLAN_TIMEOUT = 45  # Maximum time to follow path


def wrap_angle(angle):
    """
    Wrap angle to within the range [-2pi, 2pi].

    Arguments:
        angle: The angle to wrap in radians.

    Returns:
        The input angle wrapped within the range [-2pi, 2pi].
    """
    while(angle > math.pi):
        angle -= 2*math.pi
    while(angle <= -math.pi):
        angle += 2*math.pi
    return angle


def follow_path(path, goal_x, goal_y, robot, timeout=45):
    """
    Follow a path of (x, y) waypoints using simple navigation.
    Refactored from C++ p3 project to Python.
    
    Args:
        path: List of [x, y] waypoints
        goal_x: Final goal x position
        goal_y: Final goal y position
        robot: MBot object
        timeout: Maximum time in seconds
    
    Returns:
        True if goal reached, False if timeout
    """
    if not path:
        print("ERROR: Empty path!")
        return False
    
    print(f"INFO: Following path with {len(path)} waypoints...")
    start_time = time.time()
    
    # Start from waypoint 0 (or skip first few if too close)
    current_idx = 0
    
    # Skip waypoints that are too close to current position
    p_x, p_y, p_t = robot.read_slam_pose()
    while current_idx < len(path) - 1:
        wx, wy = path[current_idx]
        dist = math.sqrt((wx - p_x)**2 + (wy - p_y)**2)
        if dist > 0.15:  # Only target waypoints >15cm away
            break
        current_idx += 1
    
    while current_idx < len(path):
        # Check timeout
        if time.time() - start_time > timeout:
            print("ERROR: Path following timeout!")
            robot.stop()
            return False
        
        # Get current pose
        p_x, p_y, p_t = robot.read_slam_pose()
        
        # Get target waypoint
        target_x, target_y = path[current_idx]
        
        # Calculate distance and angle to target
        dx = target_x - p_x
        dy = target_y - p_y
        dist_to_waypoint = math.sqrt(dx**2 + dy**2)
        
        # If close to waypoint, move to next one
        if dist_to_waypoint < 0.15:  # Within 15cm
            current_idx += 1
            if current_idx >= len(path):
                # Check if at final goal
                dist_to_goal = math.sqrt((goal_x - p_x)**2 + (goal_y - p_y)**2)
                if dist_to_goal < POSE_ERROR:
                    print("INFO: Reached goal!")
                    robot.stop()
                    return True
            continue
        
        # Calculate desired heading
        desired_heading = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_heading - p_t)
        
        # If heading error is large, turn in place first
        if abs(heading_error) > 0.5:  # More than ~30 degrees
            turn_speed = TURN_KP * heading_error
            turn_speed = max(min(turn_speed, MAX_TURN_SPEED), -MAX_TURN_SPEED)
            robot.drive(0, 0, turn_speed)
        else:
            # Drive forward while correcting heading
            speed = min(0.25, dist_to_waypoint * 2)  # Scale speed with distance
            
            # Drive forward in robot frame with turning correction
            # vx is forward speed (always positive), turn corrects heading
            turn = 2.0 * heading_error
            
            robot.drive(speed, 0, turn)
        
        time.sleep(0.05)
    
    # Final check - make sure we are at goal
    p_x, p_y, p_t = robot.read_slam_pose()
    dist_to_goal = math.sqrt((goal_x - p_x)**2 + (goal_y - p_y)**2)
    
    if dist_to_goal < 0.2:  # Within 20cm is good enough
        robot.stop()
        return True
    else:
        robot.stop()
        return False


def plan_to_pose(x, y, robot):
    """
    Plan a path and drive the robot to goal [x, y] position using BFS.
    Refactored from C++ p3 project to Python.

    Arguments:
        x: Goal x position in meters in map frame.
        y: Goal y position in meters in map frame.
        robot: Mbot object.
    Returns:
        True if successful, False otherwise
    """
    # Show current position for debugging
    p_x, p_y, p_t = robot.read_slam_pose()
    current_dist = ((p_x - x)**2 + (p_y - y)**2)**0.5
    print(f"INFO: Planning to pose ({x}, {y})...")
    print(f"INFO: Current robot position: ({p_x:.3f}, {p_y:.3f}, {p_t:.3f})")
    print(f"INFO: Distance to goal: {current_dist:.3f}m")
    
    # Check if already at goal
    if current_dist < POSE_ERROR:
        print("INFO: Already at goal!")
        return True

    # Load the map
    graph = GridGraph()
    if not load_from_file(MAP_PATH, graph):
        print(f"ERROR: Failed to load map from {MAP_PATH}")
        return
    
    # Convert positions to cells
    start = pos_to_cell(p_x, p_y, graph)
    goal = pos_to_cell(x, y, graph)
    
    print(f"INFO: Start cell: ({start.i}, {start.j})")
    print(f"INFO: Goal cell: ({goal.i}, {goal.j})")
    
    # Bounds check
    if not (0 <= start.i < graph.width and 0 <= start.j < graph.height):
        print(f"ERROR: Start position ({p_x}, {p_y}) is outside map bounds!")
        return False
    
    if not (0 <= goal.i < graph.width and 0 <= goal.j < graph.height):
        print(f"ERROR: Goal position ({x}, {y}) is outside map bounds!")
        return False

    # Run BFS to find path
    print("Running BFS...")
    path = breadth_first_search(graph, start, goal)
    
    pose_path = []
    for cell in path:
        x_coord = (cell.i + 0.5) * graph.meters_per_cell + graph.origin_x
        y_coord = (cell.j + 0.5) * graph.meters_per_cell + graph.origin_y
        pose_path.append([x_coord, y_coord])
        
    # Follow the path
    success = follow_path(pose_path, x, y, robot, timeout=PLAN_TIMEOUT)
    
    if success:
        print("INFO: Successfully reached goal!")
        return True
    else:
        print("ERROR: Failed to reach goal")
        return False


def turn_to_theta(theta, robot):
    """
    Control to the angle theta using p control.

    Arguments:
        theta: Goal angle in radians.
        robot: Mbot object.
    Returns:
        True if successful, False if timeout
    """
    print(f"INFO: Turning to theta {theta} radians...")

    start_time = time.time()
    TURN_TIMEOUT = 10

    p_x,p_y, p_t = robot.read_slam_pose()

    while(abs(wrap_angle(theta - p_t)) > TURN_ERROR):
        # Check timeout
        if time.time() - start_time > TURN_TIMEOUT:
            print(f"WARNING: Turn timeout - current: {p_t:.3f}, target: {theta:.3f}")
            robot.stop()
            return False

        error = wrap_angle(theta - p_t)
        p = TURN_KP * error

        if p > MAX_TURN_SPEED:
            robot.drive(0, 0, MAX_TURN_SPEED)
        elif p < -MAX_TURN_SPEED:
            robot.drive(0, 0, -MAX_TURN_SPEED)
        else:
            robot.drive(0, 0, p)
        
        time.sleep(0.1)
        p_x, p_y, p_t = robot.read_slam_pose()

    robot.stop()
    print("Finished turning!")
    return True