````markdown
## <img src="https://github.com/decargroup/miluv/blob/gh-pages/assets/decar_logo.png?raw=true" alt="DECAR Logo" width="14"/> DECAR &mdash; MILUV devkit
Welcome to the MILUV devkit page. This Python devkit provides useful functions and examples to accompany the MILUV dataset. To begin using this devkit, clone or download and extract the repository.
![](https://github.com/decargroup/miluv/blob/gh-pages/assets/banner_image.jpg?raw=true)

## Table of Contents
- [Changelog](#changelog)
- [Devkit setup and installation](#devkit-setup-and-installation)
- [Getting started with MILUV](#getting-started-with-MILUV)
    - [Setting up the dataset](#setting-up-the-dataset)
    - [Examples](#examples)
- [Wiki](#wiki)
- [License](#license)

## Changelog
03-07-2024: MILUV devkit v1.0.0 released.

## Devkit setup and installation
The devkit requires Python 3.8 or greater. To install the devkit and its dependencies, run
```
pip3 install .
``` 
inside the devkit's root directory (~/path/to/project/MILUV). 

Alternatively, run
```
pip3 install -e .
```
inside the devkit's root directory, which installs the package in-place, allowing you make changes to the code without having to reinstall every time. 

For a list of all dependencies, refer to ``requirements.txt`` in the repository's root directory.

To ensure installation was completed without any errors, test the code by running
```
pytest
```    
in the root directory.

## Getting started with MILUV
### Setting up the dataset
To get started, download the MILUV dataset. By default, the devkit expects the data for each experiment is present in **/miluv/data/EXP_NUMBER**, where EXP_NUMBER is the number of the experiment.

If you wish to change the default data directory, be sure to modify the data directory in the devkit code.

### Examples
Using the MILUV devkit, retrieving sensor data by timestamp from experiment ``1c`` can be implemented as:
```py
from miluv.data import DataLoader
import numpy as np

mv = DataLoader(
    "default_3_random_0",
    height=False,
)

timestamps = np.arange(0, 10, 1)  # Time in s

data_at_timestamps = mv.data_from_timestamps(timestamps)
```

This example can be made elaborate by selecting specific robots and sensors to fetch from at the given timestamps.
```py
from miluv.data import DataLoader
import numpy as np

mv = DataLoader(
    "default_3_random_0",
    height=False,
)

timestamps = np.arange(0, 10, 1)  # Time in s

robots = ["ifo001", "ifo002"]  # We are leaving out ifo003
sensors = ["imu_px4", "imu_cam"]  # Fetching just the imu data

data_at_timestamps = mv.data_from_timestamps(
    timestamps,
    robots,
    sensors,
)
```

## Wiki
For more information regarding the MILUV development kit, please refer to the [documentation](https://decargroup.github.io/miluv/).

## License
This development kit is distributed under the MIT License.

````
````markdown
# MILUV Dataset Guide

This document provides a comprehensive guide to the MILUV dataset, detailing its structure, contents, and usage.

## Dataset Overview

The MILUV dataset is designed for research in multi-robot systems, localization, and sensor fusion. It contains a rich collection of sensor data from multiple unmanned aerial vehicles (UAVs) operating in a controlled environment.

## Data Structure

The dataset is organized into experiments, each with a unique identifier. Within each experiment, data is further categorized by robot and sensor type.

### Directory Layout

The typical directory structure for an experiment is as follows:

```
<experiment_name>/
├── <robot_id>/
│   ├── <sensor_name>.csv
│   └── ...
└── ...
```

- `<experiment_name>`: The name of the experiment (e.g., `default_1_random3_0`).
- `<robot_id>`: The identifier for each robot (e.g., `ifo001`, `ifo002`).
- `<sensor_name>.csv`: The CSV file containing data from a specific sensor.

### Sensor Data

The dataset includes data from a variety of sensors, including:

- **IMU**: Inertial Measurement Unit data (accelerometer and gyroscope).
- **UWB**: Ultra-Wideband ranging data.
- **Camera**: Image data for visual-inertial odometry.
- **Height Sensor**: Laser rangefinder for altitude measurements.
- **Mocap**: Ground truth pose data from a motion capture system.

## Using the Data

The `miluv` Python package provides tools for loading and processing the dataset. Refer to the [Data Loading example](examples/dataloading.md) for a practical guide on how to use the `DataLoader` class.

## Citing the Dataset

If you use the MILUV dataset in your research, please cite our work. The citation details will be provided here.

````
````markdown
# Getting Started with MILUV

This guide will walk you through the initial steps to get up and running with the MILUV devkit and dataset.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- `pip` (Python package installer)

## 1. Clone the Repository

First, clone the MILUV devkit repository from GitHub:

```bash
git clone https://github.com/decargroup/miluv.git
cd miluv
```

## 2. Install Dependencies

Install the necessary Python packages by running the following command in the root directory of the repository:

```bash
pip install .
```

This command reads the `setup.py` and `requirements.txt` files and installs the `miluv` package along with all its dependencies. For development, you might prefer an editable install:

```bash
pip install -e .
```

## 3. Download the Dataset

Download the MILUV dataset from the provided source. By default, the devkit expects the data to be located in a `data` directory within the project structure.

```
miluv/
├── data/
│   ├── <experiment_name>/
│   │   └── ...
│   └── ...
├── miluv/
│   └── ...
└── ...
```

If you place the data elsewhere, you will need to specify the path when using the `DataLoader`.

## 4. Run an Example

To verify that everything is set up correctly, you can run one of the provided examples. For instance, to run the data extraction example:

```bash
python examples/extract_data.py
```

This script demonstrates how to load data from an experiment and query it by timestamps.

## 5. Explore the Documentation

For more in-depth information, explore the rest of the documentation:

- **[Data Loading](examples/dataloading.md)**: Learn how to use the `DataLoader` to access sensor data.
- **[VINS Example](examples/vins.md)**: See how to work with Visual-Inertial Navigation System data.
- **[EKF Examples](examples/ekf/index.md)**: Dive into the Extended Kalman Filter implementations.

You are now ready to start using the MILUV devkit for your own research and development!
````
````markdown
# MILUV Documentation

Welcome to the official documentation for the MILUV dataset and development kit.

## Introduction

MILUV is a comprehensive dataset for multi-robot research, providing a wide array of sensor data from synchronized unmanned aerial vehicles (UAVs). This documentation serves as a central resource for understanding and utilizing the dataset and its accompanying tools.

## Table of Contents

- **[Getting Started](gettingstarted.md)**
  - A step-by-step guide to setting up the devkit and dataset.

- **[Dataset Guide](data.md)**
  - Detailed information about the structure and contents of the MILUV dataset.

- **[Examples](examples/index.md)**
  - Practical examples demonstrating how to use the devkit for various tasks.

- **[API Reference]**
  - (Coming soon) Detailed documentation of the `miluv` Python package API.

## Citing MILUV

If you use the MILUV dataset or devkit in your research, we kindly ask you to cite our work. The official citation will be provided here upon publication.

## Contributing

We welcome contributions to the MILUV devkit. If you have improvements, bug fixes, or new examples, please feel free to submit a pull request on our [GitHub repository](https://github.com/decargroup/miluv).

## License

The MILUV devkit is released under the MIT License. See the `LICENSE` file for more details.
````
````markdown
# AprilTag Detection

This example demonstrates how to use the `miluv` devkit to detect AprilTags from the camera data in the dataset.

## Overview

AprilTags are fiducial markers that can be used for camera calibration, object tracking, and localization. The MILUV dataset includes camera images that may contain AprilTags, and this guide shows how to process these images.

## Dependencies

To run this example, you will need an AprilTag detection library for Python. A popular choice is `pupil-apriltags`. You can install it via pip:

```bash
pip install pupil-apriltags
```

## Example Code

The following script loads an experiment, extracts camera images, and runs an AprilTag detector on each image.

```python
import cv2
from pupil_apriltags import Detector
from miluv.data import DataLoader

# Initialize the AprilTag detector
detector = Detector(families='tag36h11')

# Load the dataset
exp_name = "default_1_random3_0"  # Use an experiment with camera data
miluv_loader = DataLoader(exp_name, cam="realsense")

# Get camera data for a specific robot
cam_data = miluv_loader.data["ifo001"]["cam_realsense"]

# Iterate through the images
for index, row in cam_data.iterrows():
    image_path = row['image_path']  # Assuming the path is in the dataframe
    
    # Read the image
    # Note: You might need to construct the full path to the image
    full_image_path = f"{miluv_loader.data_dir}/{exp_name}/{image_path}"
    image = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Warning: Could not read image at {full_image_path}")
        continue

    # Detect AprilTags
    tags = detector.detect(image)
    
    if tags:
        print(f"Found {len(tags)} tags in {image_path}:")
        for tag in tags:
            print(f"  - Tag ID: {tag.tag_id}, Center: {tag.center}")

    # Optional: Visualize the detections
    # for tag in tags:
    #     for idx in range(len(tag.corners)):
    #         cv2.line(image, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (255, 0, 0), 2)
    # cv2.imshow('Detections', image)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# cv2.destroyAllWindows()
```

## Explanation

1.  **Import Libraries**: We import `cv2` for image processing, `Detector` from `pupil-apriltags`, and `DataLoader` from `miluv`.
2.  **Initialize Detector**: We create an instance of the AprilTag detector. The `families` argument specifies the type of AprilTags to look for.
3.  **Load Data**: We use `DataLoader` to load an experiment, making sure to specify a camera (`cam="realsense"`).
4.  **Iterate and Detect**: The code loops through the camera data, reads each image, and passes it to the `detector.detect()` method.
5.  **Print Results**: If any tags are found, their ID and location are printed to the console.

This example provides a basic framework. You can extend it to perform pose estimation, use the detections for SLAM, or other applications.
````
````markdown
# Data Loading with MILUV

This guide provides a detailed walkthrough of how to use the `DataLoader` class in the `miluv` devkit to access the sensor data from the MILUV dataset.

## The `DataLoader` Class

The `DataLoader` is the primary interface for interacting with the dataset. It handles loading data from CSV files, creating interpolation splines for continuous data, and providing convenient ways to query the data.

### Initialization

To start, you need to create an instance of the `DataLoader`. The main argument is the experiment name.

```python
from miluv.data import DataLoader

# Load data for a specific experiment
exp_name = "default_1_random3_0"
miluv_loader = DataLoader(exp_name)
```

By default, the `DataLoader` loads all available sensor data for all robots in the experiment. You can customize this by passing additional arguments:

```python
# Load only IMU and UWB data
miluv_loader = DataLoader(exp_name, imu="px4", uwb=True, cam=None)

# Load data for a specific robot
miluv_loader = DataLoader(exp_name, robots=["ifo001"])
```

### Accessing Data

The loaded data is stored in the `miluv_loader.data` attribute, which is a nested dictionary. The structure is `data[robot_id][sensor_name]`.

```python
# Access the PX4 IMU data for robot ifo001
imu_data = miluv_loader.data["ifo001"]["imu_px4"]

# Access the UWB range data for robot ifo002
uwb_data = miluv_loader.data["ifo002"]["uwb_range"]

# `imu_data` and `uwb_data` are pandas DataFrames
print(imu_data.head())
```

### Interpolated Data

For certain data types like motion capture (mocap), the `DataLoader` automatically creates interpolation splines. This allows you to query the data at any timestamp, not just the ones present in the original file.

```python
import numpy as np

# Get the mocap position spline for robot ifo001
mocap_pos_spline = miluv_loader.data["ifo001"]["mocap_pos"]

# Query the position at a specific time
time_t = 10.5  # seconds
position_at_t = mocap_pos_spline(time_t)
print(f"Position at t={time_t}: {position_at_t}")

# Query positions at multiple timestamps
timestamps = np.linspace(5, 15, 100)
positions = mocap_pos_spline(timestamps)
```

You can also get derivatives from the splines:

```python
# Get the velocity (first derivative) at time_t
velocity_at_t = mocap_pos_spline.derivative(nu=1)(time_t)
print(f"Velocity at t={time_t}: {velocity_at_t}")
```

## Querying by Timestamp

A common task is to get a snapshot of all sensor data at a specific time or a range of times. The `query_by_timestamps` method is designed for this.

```python
# Define the timestamps you are interested in
query_times = np.array([5.0, 5.1, 5.2])

# Query the data
queried_data = miluv_loader.query_by_timestamps(query_times)

# The result is a dictionary, similar to the main data object
# but containing data only at the specified timestamps
imu_at_query_times = queried_data["ifo001"]["imu_px4"]
print(imu_at_query_times)
```

This method performs interpolation for continuous data and finds the nearest available measurement for discrete data, making it easy to align data from different sensors.
````
````markdown
# MILUV Examples

This section provides a collection of examples to help you get started with the MILUV devkit. Each example focuses on a specific task or feature of the dataset.

## Table of Contents

- **[Data Loading](dataloading.md)**
  - Learn the basics of using the `DataLoader` to access and query sensor data.

- **[AprilTag Detection](apriltag.md)**
  - A guide on how to process camera images to detect AprilTag fiducial markers.

- **[Visualize IMU Data](visualizeimu.md)**
  - An example of how to plot and analyze Inertial Measurement Unit (IMU) data.

- **[VINS Integration](vins.md)**
  - Shows how to work with the Visual-Inertial Navigation System (VINS) data provided in the dataset.

- **[LoS Classification](losclassification.md)**
  - An example related to Line-of-Sight (LoS) and Non-Line-of-Sight (NLoS) classification for UWB signals.

- **[Extended Kalman Filter (EKF)](ekf/index.md)**
  - A set of more advanced examples implementing EKFs for state estimation using different sensor combinations.

## Running the Examples

The example scripts are located in the `examples/` directory of the repository. You can run them directly from your terminal. For example:

```bash
python examples/visualize_imu.py
```

Make sure you have installed the devkit and downloaded the dataset as described in the [Getting Started](gettingstarted.md) guide.
````
````markdown
# Line-of-Sight (LoS) Classification

This example demonstrates how to approach the problem of Line-of-Sight (LoS) and Non-Line-of-Sight (NLoS) classification for Ultra-Wideband (UWB) signals using the MILUV dataset.

## Overview

In UWB-based localization, the accuracy of range measurements can be significantly degraded when the direct path between two devices is obstructed. This is known as an NLoS condition. Identifying and mitigating NLoS measurements is crucial for robust positioning. The MILUV dataset provides Channel Impulse Response (CIR) data, which can be used to train classifiers for this purpose.

## The Data

The key data for this task is the UWB Channel Impulse Response (CIR). The CIR represents the multipath profile of the signal as it travels from the transmitter to the receiver.

- **LoS Condition**: The CIR typically shows a strong, sharp first peak, as the signal arrives unobstructed.
- **NLoS Condition**: The first peak is often attenuated and delayed, and the overall energy of the CIR might be spread out over time due to reflections and diffractions.

The `DataLoader` can be configured to load CIR data:

```python
from miluv.data import DataLoader

# Make sure to enable UWB and CIR loading
miluv_loader = DataLoader("default_1_random3_0", uwb=True, cir=True)

# Access the CIR data
cir_data = miluv_loader.data["ifo001"]["uwb_cir"]
print(cir_data.head())
```

## Example: A Simple LoS/NLoS Classifier

Here's a conceptual example of how you might build a simple classifier based on CIR features.

```python
import numpy as np
import pandas as pd
from miluv.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 1. Load Data
exp_name = "default_1_random3_0"
# Note: Ground truth LoS/NLoS labels are needed for training.
# These might be generated based on mocap data and a map of the environment.
# For this example, we'll assume a 'los' column exists in the CIR data.
miluv_loader = DataLoader(exp_name, uwb=True, cir=True)
cir_data = miluv_loader.data["ifo001"]["uwb_cir"]

# This is a placeholder - you would need to generate actual labels
cir_data['los'] = np.random.randint(0, 2, cir_data.shape[0]) 

# 2. Feature Engineering
# Extract features from the CIR. The CIR data is often stored as a string of numbers.
def extract_cir_features(row):
    # CIR is stored in the 'cir' column as a space-separated string
    cir_values = np.fromstring(row['cir'], sep=' ')
    
    # Example features
    max_power = np.max(cir_values)
    total_power = np.sum(cir_values)
    peak_to_average_ratio = max_power / (total_power / len(cir_values))
    
    return pd.Series([max_power, total_power, peak_to_average_ratio])

features = cir_data.apply(extract_cir_features, axis=1)
features.columns = ['max_power', 'total_power', 'papr']
labels = cir_data['los']

# 3. Train Classifier
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# 4. Evaluate
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

```

### Explanation

1.  **Load Data**: We load the CIR data. A crucial step, not fully implemented here, is generating ground truth LoS/NLoS labels. This is typically done by using the mocap data to check for geometric obstructions between the UWB transmitter and receiver.
2.  **Feature Engineering**: We define a function to extract meaningful features from the raw CIR data. Features like maximum power, total power, and the ratio of the peak to the average power are common choices.
3.  **Train Classifier**: We use a standard machine learning workflow. The data is split into training and testing sets, and a `RandomForestClassifier` is trained on the features.
4.  **Evaluate**: The performance of the classifier is evaluated on the test set.

This example provides a starting point. More sophisticated features and models, such as Convolutional Neural Networks (CNNs) applied directly to the CIR, can achieve higher performance.
````
````markdown
# VINS Integration

This document explains how to work with the Visual-Inertial Navigation System (VINS) data included in the MILUV dataset.

## What is VINS?

VINS is a technique that fuses data from a camera and an Inertial Measurement Unit (IMU) to estimate the motion of a device. It is a popular method for robot localization and navigation, especially in GPS-denied environments.

The MILUV dataset provides pre-processed VINS output, which can be used as a sensor measurement for higher-level state estimation tasks, such as in an Extended Kalman Filter (EKF).

## Loading VINS Data

The VINS data is not loaded by the `DataLoader` by default. Instead, a utility function `load_vins` is provided in `miluv.utils`.

```python
from miluv import utils

# Define the experiment and robot
exp_name = "default_1_random3_0"
robot_id = "ifo001"

# Load the VINS data
# loop=False: Use the VINS output before loop closure corrections
# postprocessed=True: Use the data that has been aligned with the ground truth frame
vins_data = utils.load_vins(exp_name, robot_id, loop=False, postprocessed=True)

# The result is a pandas DataFrame
print(vins_data.head())
```

### VINS DataFrame Columns

The `vins_data` DataFrame typically contains the following columns:

-   `timestamp`: The timestamp of the estimate.
-   `pose.position.x`, `pose.position.y`, `pose.position.z`: The estimated position.
-   `pose.orientation.x`, `pose.orientation.y`, `pose.orientation.z`, `pose.orientation.w`: The estimated orientation as a quaternion.
-   `twist.linear.x`, `twist.linear.y`, `twist.linear.z`: The estimated linear velocity.
-   `twist.angular.x`, `twist.angular.y`, `twist.angular.z`: The estimated angular velocity.

## Using VINS Data in an EKF

The VINS output, particularly the velocity estimates, can serve as the process model input for an EKF. This is demonstrated in the [VINS EKF example](ekf/se3_one_robot.md).

The general idea is:

1.  **State**: The EKF state is the robot's pose, e.g., $\mathbf{T} \in SE(3)$.
2.  **Process Model**: The VINS velocity estimates $(\mathbf{v}, \boldsymbol{\omega})$ are used to propagate the state forward in time.
    $$
    \dot{\mathbf{T}} = \mathbf{T} \begin{bmatrix} \boldsymbol{\omega} \\ \mathbf{v} \end{bmatrix}^{\wedge}
    $$
3.  **Correction Step**: Other sensors, like UWB for range measurements or a height sensor, are used to correct the state estimate.

### Coordinate Frame Alignment

A critical aspect of using VINS data is ensuring that all data is in a consistent coordinate frame. The `postprocessed=True` option in `load_vins` provides data that has been aligned to the global motion capture frame, which simplifies integration with other sensors. However, for the process model, you often need the velocity in the robot's body frame. This requires transforming the velocity from the global frame to the body frame using the estimated orientation.
````
````markdown
# Visualize IMU Data

This example demonstrates how to load Inertial Measurement Unit (IMU) data from the MILUV dataset and create plots to visualize it.

## Overview

Visualizing sensor data is a crucial first step in any robotics or sensor fusion project. It helps in understanding the data characteristics, identifying noise, and spotting potential issues. This guide will show you how to plot accelerometer and gyroscope data.

## Dependencies

You will need `matplotlib` for plotting. If you don't have it installed, you can install it via pip:

```bash
pip install matplotlib
```

## Example Code

The following script loads IMU data for a robot and plots the 3-axis accelerometer and gyroscope readings over time.

```python
import matplotlib.pyplot as plt
from miluv.data import DataLoader

# 1. Load the data
exp_name = "default_1_random3_0"
# We specify the IMU we want to load, e.g., "px4"
miluv_loader = DataLoader(exp_name, imu="px4")

# Get the IMU data for a specific robot
robot_id = "ifo001"
imu_data = miluv_loader.data[robot_id]["imu_px4"]

# 2. Prepare data for plotting
timestamps = imu_data['timestamp']
accel_x = imu_data['linear_acceleration.x']
accel_y = imu_data['linear_acceleration.y']
accel_z = imu_data['linear_acceleration.z']
gyro_x = imu_data['angular_velocity.x']
gyro_y = imu_data['angular_velocity.y']
gyro_z = imu_data['angular_velocity.z']

# 3. Create plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot accelerometer data
ax1.plot(timestamps, accel_x, label='X')
ax1.plot(timestamps, accel_y, label='Y')
ax1.plot(timestamps, accel_z, label='Z')
ax1.set_title(f'Accelerometer Data for {robot_id}')
ax1.set_ylabel('Acceleration (m/s^2)')
ax1.legend()
ax1.grid(True)

# Plot gyroscope data
ax2.plot(timestamps, gyro_x, label='X')
ax2.plot(timestamps, gyro_y, label='Y')
ax2.plot(timestamps, gyro_z, label='Z')
ax2.set_title(f'Gyroscope Data for {robot_id}')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Explanation

1.  **Load Data**: We initialize `DataLoader` for a specific experiment and tell it to load the `px4` IMU data. We then access the corresponding pandas DataFrame.
2.  **Prepare Data**: We extract the columns we want to plot from the DataFrame. The `timestamp` column will be our x-axis, and the sensor readings will be our y-axes.
3.  **Create Plots**: We use `matplotlib.pyplot.subplots` to create a figure with two subplots (one for the accelerometer, one for the gyroscope). We then use the `plot` function for each axis to create the line plots. Finally, we add titles, labels, and a legend to make the plots informative, and `plt.show()` displays the plot.

This script provides a template for visualizing time-series sensor data. You can adapt it to plot other data from the MILUV dataset, such as magnetometer readings, UWB ranges, or height sensor data.
````
````markdown
---
title: Extended Kalman Filter
parent: Examples
has_children: true
nav_order: 6
---

# Extended Kalman Filter (EKF) Examples

This section provides a set of advanced examples that implement Extended Kalman Filters (EKFs) for robot state estimation. These examples demonstrate how to fuse different combinations of sensors from the MILUV dataset to achieve accurate localization.

## Overview

The EKF is a popular algorithm for state estimation in nonlinear systems. It linearizes the system dynamics and measurement models around the current state estimate to propagate and update the state and its covariance.

The examples provided here are structured to showcase different EKF formulations based on the chosen state representation and sensor inputs.

## EKF Implementations

The following EKF examples are available:

-   **[$SE(3)$ VINS - 1 robot](se3_one_robot.md)**
    -   **State**: Robot pose in $SE(3)$ (3D rotation + 3D translation).
    -   **Process Model**: VINS velocity and gyroscope data.
    -   **Measurements**: UWB range and height sensor data.
    -   This example is a good starting point for understanding the basics of Lie group EKFs.

-   **[$SE_2(3)$ IMU - 1 robot](se23_one_robot.md)**
    -   **State**: Robot pose and velocity in $SE_2(3)$ (3D rotation + 3D translation + 3D velocity).
    -   **Process Model**: IMU data (accelerometer and gyroscope).
    -   **Measurements**: UWB range and height sensor data.
    -   This is a more advanced example that includes IMU bias estimation.

-   **[$SE(3)$ VINS - 3 robots]**
    -   (Coming soon) An example of a collaborative EKF for a multi-robot system.

-   **[$SE_2(3)$ IMU - 3 robots]**
    -   (Coming soon) A collaborative EKF using IMU data for a multi-robot system.

## Structure of the Examples

Each EKF example follows a similar structure:

1.  **Introduction**: An overview of the specific EKF problem, including the state definition and sensor setup.
2.  **Data Loading**: How to load the necessary sensor and ground truth data using the `miluv` devkit.
3.  **EKF Implementation**: A walkthrough of the main EKF loop, explaining the prediction and correction steps. The detailed mathematical models and Jacobians are typically encapsulated in a separate module for clarity.
4.  **Results**: Plots and analysis showing the performance of the EKF compared to the ground truth data.

These examples are designed to be both educational and practical, providing a solid foundation for developing your own state estimation algorithms with the MILUV dataset.
````
````markdown
---
title: $SE_2(3)$ IMU - 1 robot
parent: Extended Kalman Filter
usemathjax: true
nav_order: 3
---

# $SE_2(3)$ EKF with IMU - One Robot

![The setup for the one-robot IMU EKF](https://decargroup.github.io/miluv/assets/one_robot.png)

This example shows how we can use MILUV to test out an Extended Kalman Filter (EKF) for a single robot using an Inertial Measurement Unit (IMU). The derivations here are a little bit more involved than the [VINS EKF example](https://decargroup.github.io/miluv/docs/examples/ekf/se3_one_robot.html), but we'll show that the EKF implementation is still straightforward using the MILUV devkit. Nonetheless, we suggest looking at the VINS example first before proceeding with this one. In this example, we will use the following data:

- Gyroscope and accelerometer data from the robot's PX4 IMU. 
- UWB range data between the 2 tags on the robot and the 6 anchors in the environment.
- Height data from the robot's downward-facing laser rangefinder.
- Ground truth pose data from a motion capture system to evaluate the EKF.

The state we are trying to estimate is the robot's 3D pose in the absolute frame, which is represented by

$$ \mathbf{T}_{a1} = \begin{bmatrix} \mathbf{C}_{a1} & \mathbf{v}^{1a}_a & \mathbf{r}^{1a}_a \\ \mathbf{0} & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \in SE_2(3). $$

In this example, we also estimate the gyroscope and accelerometer biases, which are represented by

$$ \boldsymbol{\beta} = \begin{bmatrix} \boldsymbol{\beta}^\text{gyr} \\ \boldsymbol{\beta}^\text{acc} \end{bmatrix} \in \mathbb{R}^6. $$

We follow the same notation convention mentioned in the paper and assume the same assumptions introduced in the [VINS EKF example](https://decargroup.github.io/miluv/docs/examples/ekf/se3_one_robot.html).

## Importing Libraries and MILUV Utilities

We start by importing the necessary libraries and utilities for this example as in the VINS example, with the only change being the EKF model we are using.

```py
import numpy as np
import pandas as pd

from miluv.data import DataLoader
import miluv.utils as utils
import examples.ekfutils.imu_one_robot_models as model
import examples.ekfutils.common as common
```

## Loading the Data

We will use the same experiment as in the VINS example, and load only ifo001's data for this example. 

```py
exp_name = "default_1_random3_0"

miluv = DataLoader(exp_name, imu = "px4", cam = None, mag = False)
data = miluv.data["ifo001"]
```

We then extract all timestamps where exteroceptive data is available.

```py
query_timestamps = np.append(
    data["uwb_range"]["timestamp"].to_numpy(), data["height"]["timestamp"].to_numpy()
)
query_timestamps = np.sort(np.unique(query_timestamps))
```

We then use this to query the IMU data at these timestamps, extracing both the accelerometer and gyroscope data.

```py
imu_at_query_timestamps = miluv.query_by_timestamps(query_timestamps, robots="ifo001", sensors="imu_px4")["ifo001"]
accel: pd.DataFrame = imu_at_query_timestamps["imu_px4"][["timestamp", "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"]]
gyro: pd.DataFrame = imu_at_query_timestamps["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
```

To be able to evaluate our EKF, we extract the ground truth pose data at these timestamps. The `DataLoader` class provides interpolated splines for the ground truth pose data, and the reason for that is that we can query the ground truth data at any timestamp and call the `derivative` method to get higher-order derivatives of the pose data. For example, here we use the first derivative of the pose data to get the linear velocity data, which is necessary to evaluate our $SE_2(3)$ EKF. We use a helper function from the `utils` module to convert the mocap pose data and its derivatives to a list of $SE_2(3)$ poses.

```py
gt_se23 = utils.get_se23_poses(
    data["mocap_quat"](query_timestamps), data["mocap_pos"].derivative(nu=1)(query_timestamps), data["mocap_pos"](query_timestamps)
)
```

Additionally, we extract the IMU biases at these timestamps to evaluate the EKF's bias estimates. These ground truth biases are provided as part of the dataset, and are computed by comparing the smoothed IMU data with the ground truth pose data.

```py
gt_bias = imu_at_query_timestamps["imu_px4"][[
    "gyro_bias.x", "gyro_bias.y", "gyro_bias.z", 
    "accel_bias.x", "accel_bias.y", "accel_bias.z"
]].to_numpy()
```

## Extended Kalman Filter

We now implement the EKF for the one-robot IMU example. In here, we will go through some of the implementation details of the EKF, but as before, the EKF implementation is in the *models* module and the main script only calls these EKF methods to avoid cluttering the main script. 

We start by initializing a variable to store the EKF state and covariance at each query timestamp for postprocessing. Given that the state is composed of a matrix component (the pose) and a vector component (the bias), we initialize two state histories to store the pose and bias components.

```py
ekf_history = {
    "pose": common.MatrixStateHistory(state_dim=5, covariance_dim=9),
    "bias": common.VectorStateHistory(state_dim=6)
}
```

We then initialize the EKF with the first ground truth pose, the anchor positions, and UWB tag moment arms. Inside the constructor of the EKF, we add noise to have some initial uncertainty in the state, and set the initial bias estimate to zero.

```py
ekf = model.EKF(gt_se23[0], miluv.anchors, miluv.tag_moment_arms)
```

The main loop of the EKF is to iterate through the query timestamps and do the prediction and correction steps, which is where the EKF magic happens and is what we will go through next.

```py
for i in range(0, len(query_timestamps)):
    # ----> TODO: Implement the EKF prediction and correction steps
```

### Prediction

The prediction step is done using the IMU data, where the gyroscope reads a biased angular velocity 

$$ \mathbf{u}^\text{gyr} = \boldsymbol{\omega}^{1a}_1 - \boldsymbol{\beta}^\text{gyr} - \mathbf{w}^\text{gyr}, $$

and the accelerometer reads a biased specific force

$$ \mathbf{u}^\text{acc} = \mathbf{a}^{1a}_1 - \boldsymbol{\beta}^\text{acc} - \mathbf{w}^\text{acc}, $$

where $\mathbf{w}^\text{gyr}$ and $\mathbf{w}^\text{acc}$ are the white noise terms for the gyroscope and accelerometer, respectively, and $\boldsymbol{\omega}^{1a}_1$ and $\mathbf{a}^{1a}_1$ are the angular velocity and linear acceleration, respectively, in the robot's body frame. In the code, we extract the IMU data at the current query timestamp as follows:

```py
for i in range(0, len(query_timestamps)):
    input = np.array([
        gyro.iloc[i]["angular_velocity.x"], gyro.iloc[i]["angular_velocity.y"], 
        gyro.iloc[i]["angular_velocity.z"], accel.iloc[i]["linear_acceleration.x"], 
        accel.iloc[i]["linear_acceleration.y"], accel.iloc[i]["linear_acceleration.z"]
    ])

    # ----> TODO: EKF prediction using the gyro and vins data
```

The subsequent derivation is a little bit involved and we skip through a lot of the details for brevity, but for a more detailed derivation of the process model, one can refer to Chapter 9 in the book [State Estimation for Robotics, Second Edition by Timothy D. Barfoot](https://www.cambridge.org/core/books/state-estimation-for-robotics/00E53274A2F1E6CC1A55CA5C3D1C9718). 

The continuous-time process model for the orientation is given by

$$ \dot{\mathbf{ C }}_{a1} = \mathbf{C}_{a1} (\boldsymbol{\omega}^{1a} _ 1)^{\wedge}, $$

where $(\cdot)^{\wedge}$ is the skew-symmetric matrix operator in that maps an element of $\mathbb{R}^3$ to the Lie algebra of $SO(3)$. Meanwhile, the continuous-time process model for the velocity is given by

$$ \dot{\mathbf{v}}^{1a}_a = \mathbf{C}_{a1} \mathbf{a}^{1a}_1 + \mathbf{g}_a, $$

where $\mathbf{g}_a$ is the gravity vector in the absolute frame. The continuous-time process model for the position is given by

$$ \dot{\mathbf{r}}^{1a}_a = \mathbf{v}^{1a}_a. $$

We can show that the continuous-time process model for the state $\mathbf{T}_{a1}$ can be written compactly as

$$ \dot{\mathbf{T}}_{a1} = \mathbf{T}_{a1} \mathbf{U} + \mathbf{G} \mathbf{T}_{a1}, $$

where 

$$ \mathbf{U} = \begin{bmatrix} (\boldsymbol{\omega}^{1a}_1)^\wedge & \mathbf{a}^{1a}_1 & \\ & & 1 \\ & & 0 \end{bmatrix} \in \mathbb{R}^{5 \times 5}, \qquad \mathbf{G} = \begin{bmatrix} \mathbf{0} & \mathbf{g}_a & \\ & & -1 \\ & & 0 \end{bmatrix} \in \mathbb{R}^{5 \times 5}. $$

To implement the prediction step in an EKF, we first **discretize** the continuous-time process model over a timestep $\Delta t$ using the matrix exponential to yield

$$
\begin{aligned}
    \mathbf{T}_{a1,k+1} &= \operatorname{exp} (\Delta t \mathbf{G}) \mathbf{T}_{a1,k} \operatorname{exp} (\Delta t \mathbf{U}) \\
    &\triangleq \mathbf{G}_k \mathbf{T}_{a1,k} \mathbf{U}_k,
\end{aligned} 
$$

where 

$$
\begin{aligned}
    \mathbf{G}_k &= \begin{bmatrix} \mathbf{1} & \Delta t \mathbf{g}_a & - \frac{1}{2} \Delta t^2 \mathbf{g}_a \\ 
                                    & 1 & - \Delta t \\ 
                                     & & 1 \end{bmatrix}, \\
    \mathbf{U}_k &= \begin{bmatrix} \operatorname{Exp} (\Delta t \boldsymbol{\omega}) 
                                            & \Delta t \mathbf{J}_l(\Delta t \boldsymbol{\omega}) \mathbf{a} 
                                            & \frac{1}{2} \Delta t^2 \mathbf{N}( \Delta t \boldsymbol{\omega}) \mathbf{a} \\
                                    & 1 & \Delta t \\ 
                                    & & 1 \end{bmatrix},
\end{aligned}
$$

where $\operatorname{Exp} (\cdot)$ is the operator that maps an element of $\mathbb{R}^3$ to $SO(3)$, $\mathbf{J}_l(\cdot)$ is the left Jacobian of $SO(3)$, and $\mathbf{N}(\cdot)$ is defined in Appendix C of [this paper](https://arxiv.org/abs/2304.03837). Note that the subscripts and superscripts have been dropped from the inputs for brevity.

By perturbing the state and inputs in a similar manner as in the VINS example, we can show that the linearized process model for the pose state is given by

$$ \delta \boldsymbol{\xi}_{k+1} = \operatorname{Ad} (\mathbf{U}_{k-1}^{-1}) \delta \boldsymbol{\xi}_k - \mathbf{L}_{k} \delta \boldsymbol{\beta}_k + \mathbf{L}_k \delta \mathbf{w}_k, $$

where $\operatorname{Ad} (\cdot)$ is the *Adjoint* matrix in $SE_2(3)$,

$$ 
\mathbf{L}_k =
    \mathscr{J} \left( 
        - \begin{bmatrix} 
        \Delta t \boldsymbol{\omega} \\ \Delta t \mathbf{a} \\ \frac{1}{2} \Delta t^2 \mathbf{J}_l(\Delta t \boldsymbol{\omega})^{-1} \mathbf{N}(\Delta t \boldsymbol{\omega}) \mathbf{a} 
        \end{bmatrix}
    \right)
    \begin{bmatrix}
        \Delta t \mathbf{1} & 0 \\
        0 & \Delta t \mathbf{1} \\
        \Delta t^3 (\frac{1}{12} \mathbf{1}^\wedge - \frac{1}{720} \Delta t^2 \mathbf{M}) & \frac{1}{2} \Delta t^2 \mathbf{J}_l(\Delta t \boldsymbol{\omega})^{-1} \mathbf{N}(\Delta t \boldsymbol{\omega})
    \end{bmatrix}, 
$$

$\mathscr{J}(\cdot)$ is the left *Jacobian* in $SE_2(3)$, and $\mathbf{M}$ is defined as 

$$ \mathbf{M} = \boldsymbol{\omega}^\wedge \boldsymbol{\omega}^\wedge \mathbf{a}^\wedge + \boldsymbol{\omega}^\wedge (\boldsymbol{\omega}^\wedge \mathbf{a})^\wedge + (\boldsymbol{\omega}^\wedge \boldsymbol{\omega}^\wedge \mathbf{a})^\wedge. $$

This summarizes the prediction step for the pose states. Meanwhile, the prediction step for the bias states is given by a random walk model

$$ \dot{\boldsymbol{\beta}} = \mathbf{w}, $$

where $\mathbf{w}$ is the white noise term for the bias states. This can be simply discretized as

$$ \boldsymbol{\beta}_{k+1} = \boldsymbol{\beta}_k + \Delta t \mathbf{w}_k, $$

and given that this is a linear model, the Jacobians are simply the identity matrix.

As before, the process model and the Jacobians are implemented in the *models* module, and as such the prediction step boils down to simply calling the `predict` method of the EKF.

```py
for i in range(0, len(query_timestamps)):
    # .....
    
    # Do an EKF prediction using the gyro and vins data
    dt = (query_timestamps[i] - query_timestamps[i - 1]) if i > 0 else 0
    ekf.predict(input, dt)
```

Also as before, we set the process model covariances using the `get_imu_noise_params()` function in the *miluv.utils* module, which reads the robot-specific IMU noise and bias parameters from the `config/imu` folder that were extracted using the [allan_variance_ros](https://github.com/ori-drs/allan_variance_ros) package.

### Correction

The correction models for the UWB range and height data are almost identical to the [VINS EKF example](https://decargroup.github.io/miluv/docs/examples/ekf/se3_one_robot.html), so we will skip through this section. The only difference for the UWB range is that $\boldsymbol{\Pi}$ and $\mathbf{\tilde{r}}_{1}^{\tau_1 1}$ are defined as

$$ \boldsymbol{\Pi} = \begin{bmatrix} \mathbf{1}_3 & \mathbf{0}_{3 \times 2} \end{bmatrix} \in \mathbb{R}^{3 \times 5}, \qquad \mathbf{\tilde{r}}_{1}^{\tau_1 1} = \begin{bmatrix} \mathbf{r}_1^{\tau_1 1} \\ 0 \\ 1 \end{bmatrix} \in \mathbb{R}^5, $$

and the $\odot$ operator used in the Jacobian is the *odot* operator in $SE_2(3)$.

```py
# Iterate through the query timestamps
for i in range(0, len(query_timestamps)):
    # .....

    # Check if range data is available at this query timestamp, and do an EKF correction
    range_idx = np.where(data["uwb_range"]["timestamp"] == query_timestamps[i])[0]
    if len(range_idx) > 0:
        range_data = data["uwb_range"].iloc[range_idx]
        ekf.correct({
            "range": float(range_data["range"].iloc[0]),
            "to_id": int(range_data["to_id"].iloc[0]),
            "from_id": int(range_data["from_id"].iloc[0])
        })
```

Meanwhile, the only difference for the height data is that $\mathbf{a}$ and $\mathbf{b}$ are defined as

$$ \mathbf{a} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}, \qquad \mathbf{b} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}. $$

```py
for i in range(0, len(query_timestamps)):
    # .....

    # Check if height data is available at this query timestamp, and do an EKF correction
    height_idx = np.where(data["height"]["timestamp"] == query_timestamps[i])[0]
    if len(height_idx) > 0:
        height_data = data["height"].iloc[height_idx]
        ekf.correct({"height": float(height_data["range"].iloc[0])})
```

Lastly, we store the EKF state and covariance at this query timestamp for postprocessing.

```py
for i in range(0, len(query_timestamps)):
    # .....

    # Store the EKF state and covariance at this query timestamp
    ekf_history["pose"].add(query_timestamps[i], ekf.pose, ekf.pose_covariance)
    ekf_history["bias"].add(query_timestamps[i], ekf.bias, ekf.bias_covariance)
```

## Results

We can now evaluate the EKF using the ground truth data and plot the results. We first evaluate the EKF using the ground truth data and the EKF history, using the example-specific evaluation functions in the *models* module. 

```py
analysis = model.EvaluateEKF(gt_se23, gt_bias, ekf_history, exp_name)
```

Lastly, we call the following functions to plot the results and save the results to disk, and we are done!

```py
analysis.plot_error()
analysis.plot_poses()
analysis.plot_bias_error()
analysis.save_results()
```

![IMU EKF Pose Plot for Experiment #13](https://decargroup.github.io/miluv/assets/ekf_imu/13_poses.png) | ![IMU EKF Error Plot for Experiment #13](https://decargroup.github.io/miluv/assets/ekf_imu/13_error.png)


````
````markdown
---
title: $SE(3)$ VINS - 1 robot
parent: Extended Kalman Filter
usemathjax: true
nav_order: 1
---

# $SE(3)$ EKF with VINS - One Robot

## Overview

![The setup for the one-robot VINS EKF](https://decargroup.github.io/miluv/assets/one_robot.png)

This example shows he we can use MILUV to test out an Extended Kalman Filter (EKF) for a single robot using Visual-Inertial Navigation System (VINS) data. In this example, we will use the following data:

- Gyroscope data from the robot's PX4 IMU, where we remove the gyro bias for this example. 
- VINS data, which uses the robot's camera and IMU to estimate the robot's velocity.
- UWB range data between the 2 tags on the robot and the 6 anchors in the environment.
- Height data from the robot's downward-facing laser rangefinder.
- Ground truth pose data from a motion capture system to evaluate the EKF.

The state we are trying to estimate is the robot's 3D pose in the absolute frame, which is represented by

$$ \mathbf{T}_{a1} = \begin{bmatrix} \mathbf{C}_{a1} & \mathbf{r}^{1a}_a \\ \mathbf{0} & 1 \end{bmatrix} \in SE(3). $$

We follow the same notation convention as mentioned in the paper, and the reference frame $\{ F_1 \}$ is the body-fixed reference frame of robot ifo001.

As shown in the figure above, the robot has two UWB tags, $f_1$ and $s_1$, for which we define $\tau_1 \in \\{ f_1, s_1 \\}$. The robot also has 6 anchors in the environment, which are assumed to be stationary and their positions $ \mathbf{r}^{\alpha_i a}_a \in \mathbb{R}^3 $ are known in the absolute frame as provided in `config/uwb/anchors.yaml`. Similarly, the moment arm of the tags on the robot, $ \mathbf{r}^{\tau_1 1}_1 \in \mathbb{R}^3 $, is also known and provided in `config/uwb/tags.yaml`.

## Importing Libraries and MILUV Utilities

We start by importing the necessary libraries and utilities for this example. Firstly, we import the standard libraries `numpy` and `pandas` for numerical computations and data manipulation. 

```py
import numpy as np
import pandas as pd
```

We then import the `DataLoader` class from the `miluv` package, which provides an easy way to load the MILUV dataset. This is the core of the MILUV devkit, and it provides an interface to load the sensor data and ground truth data for the experiments.

```py
from miluv.data import DataLoader
```

We also import the `utils` module from the `miluv` package, which provide utilities for Lie groups that accompany and other helper functions. 

```py
import miluv.utils as utils
```

Each EKF example is accompanied by a *models* module that contains the EKF implementation for that example to hide the implementation details from the main script. This is since the process model, measurement model, jacobians, and evaluation functions specific for the EKF example are irrelevant to showcase how MILUV can be used. Additionally, the *common* module contains utility functions that are shared across all EKF examples. We import these to the main script.

```py
import examples.ekfutils.vins_one_robot_models as model
import examples.ekfutils.common as common
```

## Loading the Data

We start by defining the experiment we want to run the EKF on. In this case, we will use experiment `default_1_random3_0`.

```py
exp_name = "default_1_random3_0"
```

We then, in one line, load all the sensor data we want for our EKF. For this example, we only care about ifo001's data, and we remove the IMU bias to simplify the EKF implementation.

```py
miluv = DataLoader(exp_name, imu = "px4", cam = None, mag = False, remove_imu_bias = True)
data = miluv.data["ifo001"]
```

Additionally, we load the VINS data for ifo001, where we set `loop = False` to avoid loading the loop-closed VINS data, and `postprocessed = True` to load the VINS data that has been aligned with the mocap reference frame.

```py
vins = utils.load_vins(exp_name, "ifo001", loop = False, postprocessed = True)
```

We then extract all timestamps where exteroceptive data is available and within the time range of the VINS data.

```py
query_timestamps = np.append(
    data["uwb_range"]["timestamp"].to_numpy(), data["height"]["timestamp"].to_numpy()
)
query_timestamps = query_timestamps[query_timestamps > vins["timestamp"].iloc[0]]
query_timestamps = query_timestamps[query_timestamps < vins["timestamp"].iloc[-1]]
query_timestamps = np.sort(np.unique(query_timestamps))
```

We then use this to query the gyro and VINS data at these timestamps.

```py
imu_at_query_timestamps = miluv.query_by_timestamps(query_timestamps, robots="ifo001", sensors="imu_px4")["ifo001"]
gyro: pd.DataFrame = imu_at_query_timestamps["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
vins_at_query_timestamps = utils.zero_order_hold(query_timestamps, vins)
```

To be able to evaluate our EKF, we extract the ground truth pose data at these timestamps. The `DataLoader` class provides interpolated splines for the ground truth pose data, which we can use to get the ground truth poses at the query timestamps. We use a helper function from the `utils` module to convert the mocap pose data to a list of $SE(3)$ poses.

```py
gt_se3 = utils.get_se3_poses(
    data["mocap_quat"](query_timestamps), data["mocap_pos"](query_timestamps)
)
```

Lastly, we convert the VINS data from the absolute (mocap) frame to the robot's body frame using the ground truth data, such that we can have interoceptive data in the robot's body frame. This is a bit of a hack to simplify this EKF implementation.

```py
vins_body_frame = common.convert_vins_velocity_to_body_frame(vins_at_query_timestamps, gt_se3)
```

## Extended Kalman Filter

We now implement the EKF for the one-robot VINS example. In here, we will go through some of the implementation details of the EKF, but this is to reiterate that the EKF implementation is in the *models* module and the main script only calls these EKF methods to avoid cluttering the main script. 

We start by initializing a variable to store the EKF state and covariance at each query timestamp for postprocessing.

```py
ekf_history = common.MatrixStateHistory(state_dim=4, covariance_dim=6)
```

We then initialize the EKF with the first ground truth pose, the anchor postions, and UWB tag moment arms. Inside the constructor of the EKF, we add noise to have some initial uncertainty in the state.

```py
ekf = model.EKF(gt_se3[0], miluv.anchors, miluv.tag_moment_arms)
```

The main loop of the EKF is to iterate through the query timestamps and do the prediction and correction steps, which is where the EKF magic happens and is what we will go through next.

```py
for i in range(0, len(query_timestamps)):
    # ----> TODO: Implement the EKF prediction and correction steps
```

### Prediction

The prediction step is done using the gyroscope and VINS data. The continuous-time process model for the orientation is given by

$$ \dot{\mathbf{ C }}_{a1} = \mathbf{C}_{a1} (\boldsymbol{\omega}^{1a} _ 1)^{\wedge}, $$

where $\boldsymbol{\omega}^{1a} _ 1$ is the angular velocity measured by the gyroscope, and $(\cdot)^{\wedge}$ is the skew-symmetric matrix operator in that maps an element of $\mathbb{R}^3$ to the Lie algebra of $SO(3)$. Meanwhile, the continuous-time process model for the position is given by

$$ \dot{\mathbf{r}}^{1a}_a = \mathbf{C}_{a1} \mathbf{v}^{1a}_1, $$

where $\mathbf{v}^{1a}_1$ is the linear velocity measured by VINS after being transformed to the robot's body frame. By defining the input vector as 

$$ \mathbf{u} = \begin{bmatrix} \boldsymbol{\omega}^{1a}_1 \\ \mathbf{v}^{1a}_1 \end{bmatrix}, $$

the continuous-time process model for the state $\mathbf{T}_{a1}$ can be written compactly as

$$ \dot{\mathbf{T}}_{a1} = \mathbf{T}_{a1} \mathbf{u}^{\wedge}, $$

where $(\cdot)^{\wedge}$ here is overloaded to represent the skew-symmetric matrix operator in that maps an element of $\mathbb{R}^6$ to the Lie algebra of $SE(3)$.

In the code, we generate the input vector $\mathbf{u}$ using the gyro and VINS data at the current query timestamp as follows:

```py
for i in range(0, len(query_timestamps)):
    input = np.array([
        gyro.iloc[i]["angular_velocity.x"], gyro.iloc[i]["angular_velocity.y"], 
        gyro.iloc[i]["angular_velocity.z"], vins_body_frame.iloc[i]["twist.linear.x"],
        vins_body_frame.iloc[i]["twist.linear.y"], vins_body_frame.iloc[i]["twist.linear.z"],
    ])

    # ----> TODO: EKF prediction using the gyro and vins data
```

To implement the prediction step in an EKF, we first **discretize** the continuous-time process model over a timestep $\Delta t$ using the matrix exponential to yield

$$ \mathbf{T}_{a1,k+1} = \mathbf{T}_{a1,k} \operatorname{Exp} (\mathbf{u}_k \Delta t), $$

where $\operatorname{Exp} (\cdot)$ is the operator that maps an element of $\mathbb{R}^6$ to $SE(3)$.

In order to use the process model in propagating the covariance of the EKF, we need to **linearize** the system to obtain the Jacobians of the process model with respect to the state and input, respectively. By perturbing the input using $ \mathbf{u} = \bar{\mathbf{u}} + \delta \mathbf{u} $ and the state using 

$$ \mathbf{T}_{a1} = \bar{\mathbf{T}}_{a1} \operatorname{Exp} (\delta \boldsymbol{\xi}) \approx \bar{\mathbf{T}}_{a1} \left( \mathbf{1} + \delta \boldsymbol{\xi}^{\wedge} \right), $$

it can be shown that the linearized process model is given by

$$ \delta \boldsymbol{\xi}_{k+1} = \operatorname{Ad} (\operatorname{Exp} (\bar{\mathbf{u}}_k \Delta t)^{-1}) \delta \boldsymbol{\xi}_k + \Delta t \boldsymbol{\mathcal{J}}_l(-\Delta t \bar{\mathbf{u}_k}) \delta \mathbf{u}_k, $$

where $\operatorname{Ad} (\cdot) : SE(3) \rightarrow \mathbb{R}^{6 \times 6}$ is the *Adjoint* matrix in $SE(3)$, and $\boldsymbol{\mathcal{J}}_l(\cdot)$ is the left Jacobian of $SE(3)$.

The process model and the Jacobians are implemented in the *models* module, and as such the prediction step boils down to simply calling the `predict` method of the EKF.

```py
for i in range(0, len(query_timestamps)):
    # .....
    
    # Do an EKF prediction using the gyro and vins data
    dt = (query_timestamps[i] - query_timestamps[i - 1]) if i > 0 else 0
    ekf.predict(input, dt)
```

Although it is not shown in the script, we set the process model covariances using the `get_imu_noise_params()` function in the *miluv.utils* module, which reads the robot-specific IMU noise parameters from the `config/imu` folder that were extracted using the [allan_variance_ros](https://github.com/ori-drs/allan_variance_ros) package.

### Correction

The correction takes a similar approach. We do correction using both the UWB range data between the anchors and the tags on the robot, and the height data from the robot's downward-facing laser rangefinder. Starting with the UWB range data, the measurement model is given by

$$ y = \lVert \mathbf{r}_a^{\alpha_i \tau_1} \rVert, $$

which can be written as

$$ y = \lVert \mathbf{r}_a^{\alpha_i a} - (\mathbf{C}_{a1} \mathbf{r}_1^{\tau_1 1} + \mathbf{r}_a^{1a}) \rVert. $$

It can be shown that this can be written as a function of the state using

$$ y = \lVert \mathbf{r}_a^{\alpha_i a} - \boldsymbol{\Pi} \mathbf{T}_{a1} \mathbf{\tilde{r}}_{1}^{\tau_1 1} \rVert $$

where 

$$ \boldsymbol{\Pi} = \begin{bmatrix} \mathbf{1}_3 & \mathbf{0}_{3 \times 1} \end{bmatrix} \in \mathbb{R}^{3 \times 4}, \qquad \mathbf{\tilde{r}}_{1}^{\tau_1 1} = \begin{bmatrix} \mathbf{r}_1^{\tau_1 1} \\ 1 \end{bmatrix} \in \mathbb{R}^4. $$

Deriving the Jacobian is a bit involved, but it can be shown that by defining a vector

$$ \boldsymbol{\nu} = \mathbf{r}_a^{\alpha_i a} - \boldsymbol{\Pi} \bar{\mathbf{T}} _ {a1} \mathbf{\tilde{r}} _ {1} ^ {\tau_1 1}, $$

the Jacobian of the measurement model with respect to the state is given by

$$ \delta y = - \frac{\boldsymbol{\nu}^\intercal}{\lVert \boldsymbol{\nu} \rVert} \boldsymbol{\Pi} \bar{\mathbf{T}}_{a1} (\mathbf{\tilde{r}}_{1}^{\tau_1 1})^\odot \delta \boldsymbol{\xi}, $$

where $(\cdot)^\odot : \mathbb{R}^4 \rightarrow \mathbb{R}^{4 \times 6}$ is the *odot* operator in $SE(3)$.

Similar to the prediction step, the correction step boils down to simply calling the `correct` method of the EKF as the measurement model and the Jacobians are implemented in the *models* module. We first check if range data is available at the current query timestamp, and if so, we do a correction using the range data.

```py
# Iterate through the query timestamps
for i in range(0, len(query_timestamps)):
    # .....

    # Check if range data is available at this query timestamp, and do an EKF correction
    range_idx = np.where(data["uwb_range"]["timestamp"] == query_timestamps[i])[0]
    if len(range_idx) > 0:
        range_data = data["uwb_range"].iloc[range_idx]
        ekf.correct({
            "range": float(range_data["range"].iloc[0]),
            "to_id": int(range_data["to_id"].iloc[0]),
            "from_id": int(range_data["from_id"].iloc[0])
        })
```

Meanwhile, the height data is given by

$$ y = \begin{bmatrix} 0 & 0 & 1 \end{bmatrix} \mathbf{r}_a^{1a}. $$

This can be written as a function of the state using

$$ y = \mathbf{a}^\intercal \mathbf{T}_{a1} \mathbf{b}, $$

where 

$$ \mathbf{a} = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix}, \qquad \mathbf{b} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 1 \end{bmatrix}. $$

The Jacobian of the measurement model with respect to the state can be derived to be

$$ \delta y = \mathbf{a}^\intercal \bar{\mathbf{T}}_{a1} \mathbf{b}^\odot \delta \boldsymbol{\xi}. $$

We then check if height data is available at the current query timestamp, and if so, we do a correction using the height data.

```py
for i in range(0, len(query_timestamps)):
    # .....

    # Check if height data is available at this query timestamp, and do an EKF correction
    height_idx = np.where(data["height"]["timestamp"] == query_timestamps[i])[0]
    if len(height_idx) > 0:
        height_data = data["height"].iloc[height_idx]
        ekf.correct({"height": float(height_data["range"].iloc[0])})
```

Lastly, we store the EKF state and covariance at this query timestamp for postprocessing.

```py
for i in range(0, len(query_timestamps)):
    # .....

    # Store the EKF state and covariance at this query timestamp
    ekf_history.add(query_timestamps[i], ekf.x, ekf.P)
```

## Results

We can now evaluate the EKF using the ground truth data and plot the results. We first evaluate the EKF using the ground truth data and the EKF history, using the example-specific evaluation functions in the *models* module. 

```py
analysis = model.EvaluateEKF(gt_se3, ekf_history, exp_name)
```

Lastly, we call the following functions to plot the results and save the results to disk, and we are done!

```py
analysis.plot_error()
analysis.plot_poses()
analysis.save_results()
```

![VINS EKF Pose Plot for Experiment #13](https://decargroup.github.io/miluv/assets/ekf_vins/13_poses.png) | ![VINS EKF Error Plot for Experiment #13](https://decargroup.github.io/miluv/assets/ekf_vins/13_error.png)

````
