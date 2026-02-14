# Autonomous Tour Guide Robot
## Machine Learning-Driven Navigation System

---

## Project Overview

This project presents an autonomous tour guide robot that  navigates through a mapped environment by reading handwritten numerical markers at designated waypoints. The system integrates computer vision, handwritten text recognition (HTR), path planning algorithms, and localization to create a fully autonomous navigation system that adapts its route based on visual input.

**Core Technologies:**
- LIDAR-based mapping and localization (SLAM)
- Camera-based visual perception
- Custom-trained machine learning model for digit recognition
- Breadth-First Search (BFS) path planning
- Real-time obstacle avoidance

---

## Motivation

I selected this project to deepen my understanding of machine learning model training and deployment on embedded systems. While I had previously worked with existing ML algorithms, I had never trained a model from scratch or integrated it into a robotic control system. Also, I enjoyed P3: Path Planning so I was excited to continue using it on a more complex, vision-guided autonomous system that showcases the integration of multiple robotic subsystems.

---

## Technical Implementation

### 1. Machine Learning Model Development

The HTR model required extensive training and refinement. I developed a neural network using scikit-learn's MLPClassifier to recognize handwritten digits from camera images captured by the robot.

**Model Architecture & Training:**
```python
# Neural Network Configuration
from sklearn.neural_network import MLPClassifier

# Multi-layer perceptron with two hidden layers
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Two hidden layers
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

# Training on preprocessed 28x28 grayscale images
# Input: 784-dimensional flattened image vector (28x28)
# Output: Digit classification
model.fit(X_train_normalized, y_train)
```

**Image Preprocessing Pipeline:**
```python
def preprocess_camera_image(camera_frame):
    """
    Convert raw camera frame to model-ready format.
    Ensures compatibility with trained model input requirements.
    """
    # Get processed image from camera handler (orange detection & cropping)
    pic = camera_handler.get_processed_image()
    
    if pic is None:
        return None
    
    # Flatten 28x28 image to 784-dimensional vector
    flat_pic = pic.flatten().reshape(1, -1)
    
    # Normalize pixel values to [0, 1] range
    # Critical for matching training data preprocessing
    norm_pic = flat_pic / 255.0
    
    return norm_pic

# Model inference
prediction = model.predict(norm_pic)
predicted_label = int(prediction[0])
```

### 2. Path Planning & Navigation

The navigation system uses BFS to find collision-free paths through the occupancy grid map.

**BFS Implementation with Collision Checking:**
```python
def breadth_first_search(graph, start, goal):
    """
    BFS pathfinding with 8-connectivity for diagonal movements.
    Returns optimal path from start to goal avoiding obstacles.
    """
    queue = deque([start_idx])
    came_from = {start_idx: None}
    graph.nodes[start_idx].queued = True
    
    while queue:
        current_idx = queue.popleft()
        graph.nodes[current_idx].visited = True
        
        # Goal check
        if current_idx == goal_idx:
            return trace_path(goal_idx, graph)
        
        # Explore neighbors (cardinal + diagonal)
        for neighbor_idx in find_neighbors(current_idx, graph):
            neighbor_node = graph.nodes[neighbor_idx]
            
            # Add to frontier if unvisited and collision-free
            if (not neighbor_node.visited and 
                not neighbor_node.queued and 
                not check_collision(neighbor_idx, graph)):
                
                neighbor_node.parent = current_idx
                neighbor_node.cost = graph.nodes[current_idx].cost + 1
                neighbor_node.queued = True
                queue.append(neighbor_idx)
    
    return []  # No path found

def check_collision(idx, graph):
    """
    Collision detection with safety padding.
    Checks 3-cell radius around robot to ensure clearance.
    """
    cell = idx_to_cell(idx, graph)
    padding = 3
    
    for di in range(-padding, padding + 1):
        for dj in range(-padding, padding + 1):
            check_i, check_j = cell.i + di, cell.j + dj
            if not is_cell_in_bounds(check_i, check_j, graph):
                return True
            check_idx = cell_to_idx(check_i, check_j, graph)
            if graph.cell_odds[check_idx] > 50:  # Obstacle threshold
                return True
    return False
```

### 3. Tour Logic & Decision Making

The main control loop integrates vision, navigation, and decision-making to execute the autonomous tour.

```python
def execute_tour(robot, model, waypoints, waypoint_map):
    """
    Main tour execution loop.
    Reads markers, makes decisions, and navigates autonomously.
    """
    camera = CameraHandler()
    next_waypoint = waypoints[4]  # Starting destination
    
    while True:
        # Navigate to waypoint
        if not plan_and_drive(robot, next_waypoint[0], 
                             next_waypoint[1], next_waypoint[2]):
            print("Navigation failed!")
            break
        
        # Capture and recognize marker (with retry logic)
        for attempt in range(3):
            image = camera.get_processed_image()
            if image is not None:
                break
            time.sleep(0.5)
        
        if image is None:
            continue  # Skip if no marker detected
        
        # ML inference
        preprocessed = preprocess_camera_image(image)
        predicted_label = int(model.predict(preprocessed)[0])
        
        # Decision logic
        if predicted_label == 0:
            # Return home and end tour
            home = waypoints[0]
            plan_and_drive(robot, home[0], home[1], home[2])
            break
        elif predicted_label in waypoint_map:
            # Navigate to next waypoint based on marker
            next_waypoint = waypoints[waypoint_map[predicted_label]]
        else:
            print(f"Unknown label: {predicted_label}")
```

---

## Significant Technical Challenges

### Challenge 1: Dependency Version Conflicts

**Problem:** The robot's existing codebase used legacy versions of NumPy (1.21.x) and scikit-learn (0.24.x), while my model was initially trained with current versions (NumPy 1.26.x, scikit-learn 1.3.x). This created version conflicts when loading the trained model.

**Solution:** I established a version-controlled development pipeline:
1. Created isolated virtual environments matching the robot's dependencies
2. Retrained the model using compatible library versions
3. Validated model performance across both environments
4. Documented exact dependency versions for reproducibility

This required training five separate models before achieving full compatibility.

### Challenge 2: Binary Executable Failure

**Problem:** The assignment specified using the P3: Path Planning binary executable, but this caused the program to hang indefinitely without robot movement.

**Solution:** I refactored my entire C++ path planning implementation to Python:
- Translated the occupancy grid representation and graph structures
- Reimplemented BFS with proper collision checking
- Developed coordinate transformation functions (`world_to_grid`, `grid_to_world`)

---

## Final Product vs. Original Vision

**Original Plan:**
- Implement A* search algorithm for optimal pathfinding
- Complex tour: Start → Blue waypoint → Green waypoint → Red waypoint → Home
- Advanced obstacle avoidance using potential fields

**Final Implementation:**
- BFS path planning (simpler but still effective)
- Simplified tour: Start → Red waypoint → Blue waypoint → Home
- Hybrid controller with basic collision avoidance

**Reason for Changes:**
Time constraints from debugging the binary executable failure and coordinate system issues forced scope reduction. The robot had difficulty with the sharp turn required to reach the green waypoint, so I simplified the route. While not as ambitious as initially envisioned, the final system successfully demonstrates autonomous navigation with vision-based decision making.

---

## Learning Outcomes

This project represents the culmination of concepts developed throughout the semester, integrating:

**From Previous Projects:**
- **Mapping & Localization:** SLAM data provides real-time pose estimation for navigation
- **Path Planning:** BFS finds collision-free paths through occupancy grids
- **Motor Control:** PID controllers and differential drive kinematics

**New Skills Developed:**
- **Machine Learning:** End-to-end model training, from data collection to deployment
- **Computer Vision:** Image preprocessing, feature extraction, and real-time inference
- **System Integration:** Combining perception, planning, and control into a cohesive autonomous system
- **Embedded ML:** Deploying trained models on resource-constrained hardware

The project demonstrates how autonomous robots integrate multiple subsystems—each individually complex—into a functioning whole. The addition of machine learning for perception represents a crucial step toward truly intelligent robotics, where robots can interpret their environment rather than simply execute preprogrammed behaviors.

---

## References

[1] scikit-learn developers, "sklearn.neural_network.MLPClassifier," scikit-learn documentation, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

[2] scikit-learn developers, "Neural network models (supervised)," scikit-learn documentation, 2024. [Online]. Available: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

[3] "Robotics 102: Introduction to AI and Programming," HelloRob.org, 2024. [Online]. Available: https://hellorob.org/

[4] MBot Documentation, "MBot Bridge API Reference," University of Michigan Robotics, 2024. [Online]. Available: https://mbot.robotics.umich.edu/

[5] NumPy developers, "NumPy User Guide," NumPy documentation, 2024. [Online]. Available: https://numpy.org/doc/stable/user/

