import sys
import os
import signal
import time
from joblib import dump, load  # To get the model to the robot
import numpy as np

from mbot_bridge.api import MBot
from utils.camera import CameraHandler
from utils.robot import plan_to_pose, turn_to_theta
from waypoint_writer import read_labels_and_waypoints


PATH_TO_MODEL = "./models/htr_new_model.joblib"

robot = MBot()

# Handle Ctrl+C 
def signal_handler(sig, frame):
    print("Stopping...")
    robot.stop()
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def main():
    cam = CameraHandler()  # Initialize the camera

    # If fails, check path.
    assert os.path.exists(PATH_TO_MODEL), f"Model file {PATH_TO_MODEL} does not exist."
    model = load(PATH_TO_MODEL)

    # Load waypoints from waypoints.txt
    labels, waypoints = read_labels_and_waypoints() 

    # dictionary mapping label -> (x, y, theta) 
    w_coordinates = {}
    for i in range(len(labels)):
        w_coordinates[labels[i]] = waypoints[i]
    
    # dictionary mapping detected label -> next destination 
    path_flow = {
        0: 4,  # Start at poster 0, go to waypoint 4
        4: 3,  # At poster 4, go to waypoint 3
        3: 0,  # At poster 3, go to waypoint 0
    }

    if len(labels) < 1:
        print("Error: No waypoints defined!")
        return

    next_point = w_coordinates[4]  # first go to waypoint 4

    print("--- Starting Tour: ---")

    while True:
        print(f"Driving to: {next_point}")

        plan_to_pose(next_point[0], next_point[1], robot)
        turn_to_theta(next_point[2], robot)
        
        # Turn to face the poster
        if not turn_to_theta(next_point[2], robot):
            print("WARNING: Failed to turn to theta")

        print(f"Arrived at waypoint: {next_point}")
        time.sleep(1)    # for camera to settle

        # Take a picture (+Retry logic)
        pic = None
        for i in range(3):    # Try 3 times to get a good picture
            pic = cam.get_processed_image()
            if pic is not None:
                break
            print("No poster... trying again.")
            time.sleep(0.5)

        # Process Picture
        if pic is None:
            print("Failed to find poster after 3 attempts.")
            cam.get_processed_image(save=True) 
            print("Skipping this waypoint...")
            continue

        # PRE-PROCESSING & PREDICT (flatten & normalize to get 784 dimensional vector)
        flat_pic = pic.flatten().reshape(1, -1)
        norm_pic = flat_pic / 255.0

        prediction = model.predict(norm_pic)
        
        # Convert prediction to int (model might return string)
        predicted_label = int(prediction[0]) 
        print(f"Detected label: {predicted_label}")

        # Tour logic

        # if 0, go home 
        if predicted_label == 0:
            print("=== Found label 0! Going Home ===")
            home = w_coordinates[0]
            
            if plan_to_pose(home[0], home[1], robot):
                turn_to_theta(home[2], robot)
                print("Tour Complete!")
            else:
                print("ERROR: Failed to navigate home")
            break
        
        elif predicted_label in path_flow:
            next_waypoint_label = path_flow[predicted_label]
            
            if next_waypoint_label not in w_coordinates:
                print(f"ERROR: Waypoint {next_waypoint_label} not in coordinates!")
                break
            
            next_point = w_coordinates[next_waypoint_label]

        else:
            print(f"WARNING: Label {predicted_label} not in waypoint map!")

if __name__ == '__main__':
    main()
