# Robot Tour Guide with Machine Learning

> **Note:** This project was originally completed as part of CSC 386: Autonomous Systems at Berea College. This is a personal copy to showcase my work.

## Overview

Autonomous robot tour guide that uses machine learning for visual recognition and navigation. The robot navigates between waypoints, identifies numbered posters using a custom-trained classifier, and makes routing decisions based on detected labels.

**For detailed technical implementation, challenges, and code samples, see [robotics-final-report.md](robotics-final-report.md).**

## Technical Stack

• **Language:** Python  
• **Hardware:** MBot robot platform  
• **Key Technologies:** Scikit-learn, OpenCV, NumPy, Joblib  
• **Core Systems:** Computer vision, MLPClassifier neural network, SLAM, BFS path planning

## Key Features

• **Machine Learning Classifier** - Custom-trained neural network for handwritten digit recognition  
• **Autonomous Navigation** - BFS path planning with collision detection and SLAM localization  
• **Computer Vision Pipeline** - Real-time image capture, preprocessing, and inference  
• **Adaptive Routing** - Dynamic path selection based on visual marker detection  
• **Robust Error Handling** - Retry logic for camera failures and navigation issues

## Highlights

✓ Trained and deployed ML model on embedded robotics platform  
✓ Integrated perception (vision), cognition (ML), and action (navigation) subsystems  
✓ Solved dependency compatibility issues across development and deployment environments  
✓ Implemented complete autonomous system with real-time decision-making

## Project Structure

```
├── robot_tour_guide.py      # Main tour guide program
├── waypoint_writer.py        # Utility to record waypoints
├── model.joblib              # Trained ML classifier
├── waypoints.txt             # Stored waypoint coordinates
├── requirements.txt          # Python dependencies
└── utils/
    ├── camera.py             # Camera interface and image processing
    ├── robot.py              # Robot navigation utilities
    └── classifier_test.py    # Classifier testing utility
```

## Setup

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

### Setting Waypoints

Run the waypoint writer while the robot is localized:
```bash
python waypoint_writer.py
```

Drive to each location using the webapp and record waypoints with appropriate labels.

### Running the Tour

Execute the main tour guide program:
```bash
python robot_tour_guide.py
```

The robot will autonomously navigate between waypoints, detect poster labels, and follow the configured tour path.

### Testing Components

Test the camera system:
```bash
python utils/camera_test.py
```

Test the classifier:
```bash
python utils/classifier_test.py
```

## Documentation

• **Technical Report:** [robotics-final-report.md](robotics-final-report.md) - Detailed implementation, code samples, and challenges  
• **Project Specifications:** [Michigan Robotics: Project 4](https://robotics102.github.io/projects/a4.html#tour_guide)

## Acknowledgments

Project completed as part of CSC 386: Autonomous Systems coursework at Berea College.
