# Repository Structure

*   **CMakeLists.txt**:
    ```cmake
    cmake_minimum_required(VERSION 3.0.2)
    project(miluv)

    find_package(catkin REQUIRED COMPONENTS
      uwb_ros
    )
    catkin_package()
    include_directories()
    ```
*   **compose.yaml**:
    ```yaml
    version: '3.9'

    services:
      miluv:
        image: miluv
        privileged: true
        network_mode: host
        environment:
          DISPLAY: $DISPLAY
          NVIDIA_VISIBLE_DEVICES: all
          NVIDIA_DRIVER_CAPABILITIES: all
        volumes:
          - ./data:/workspace/miluv/data:rw
          - ./results:/workspace/miluv/results:rw
          - /tmp/.X11-unix:/tmp/.X11-unix:rw
        stdin_open: true
        tty: true
    ```
*   **Dockerfile**:
    ```dockerfile
    # Use the official Ubuntu 24.04 as the base image
    FROM ubuntu:24.04

    # Set environment variables
    ENV LANG C.UTF-8
    ENV LC_ALL C.UTF-8

    # Set working directory
    WORKDIR /workspace

    # Install dependencies
    RUN apt-get update && apt-get install -y \
        software-properties-common \
        curl \
        gnupg2 \
        lsb-release \
        wget \
        unzip \
        build-essential \
        git \
        cmake \
        python3-pip \
        python3-venv \
        mesa-utils \
        && rm -rf /var/lib/apt/lists/*

    # Make a virtual environment and activate it
    RUN python3 -m venv /virtualenv/miluv
    RUN echo "source /virtualenv/miluv/bin/activate" >> ~/.bashrc

    # Clone the MILUV repository
    RUN git clone https://github.com/decargroup/miluv.git

    # Make symlink to build uwb_ros with UWB messages
    RUN ln -s miluv/uwb_ros .

    # Install MILUV 
    WORKDIR /workspace/miluv
    RUN /bin/bash -c "source /virtualenv/miluv/bin/activate && pip3 install csaps && pip3 install ."

    # Install some dependencies for remote visualization
    RUN apt update
    RUN /bin/bash -c "source /virtualenv/miluv/bin/activate && pip install PyQt5"
    RUN apt-get install -y libxcb-xinerama0 libxcb1 libx11-xcb1 libxrender1 libxi6 libxext6
    RUN /bin/bash -c "source /virtualenv/miluv/bin/activate && pip install opencv-python-headless"
    RUN apt-get install -y qt5-qmake qtbase5-dev qtchooser qt5-qmake-bin libqt5core5a libqt5gui5
    ENV XDG_RUNTIME_DIR=/tmp/runtime-dir

    # Expose the necessary ports
    EXPOSE 11311

    # Set entrypoint to bash so you can run commands interactively
    CMD ["/bin/bash", "-c", "source /virtualenv/miluv/bin/activate && exec /bin/bash"]
    ```
*   **LICENSE**:
    ```
    MIT License

    Copyright (c) 2024 DECAR

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    ```
*   **package.xml**:
    ```xml
    <?xml version="1.0"?>
    <package format="2">
      <name>miluv</name>
      <version>0.0.0</version>
      <description>The miluv package</description>

      <maintainer email="mohammed.shalaby@mail.mcgill.ca">Mohammed A. Shalaby</maintainer>
      <license>MIT</license>

      <buildtool_depend>catkin</buildtool_depend>

      <depend>uwb_ros</depend>

      <export></export>
    </package>
    ```
*   **README.md**:
    ```markdown
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

    If you wish to change the default data location, be sure to modify the data directory in the devkit code.

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

  
*   **requirements.txt**:
    ```
    zipp==3.18.1
    py3rosmsgs==1.18.2
    jedi==0.19.1
    setuptools-scm==8.0.4
    pyserial==3.5
    numpy==1.24.4
    packaging==24.0
    parso==0.8.3
    scipy==1.10.1
    traitlets==5.14.2
    MarkupSafe==2.1.5
    pickleshare==0.7.5
    stack-data==0.6.3
    csaps==1.1.0
    cycler==0.12.1
    pycryptodomex==3.20.0
    matplotlib-inline==0.1.6
    typing_extensions==4.10.0
    executing==2.0.1
    ptyprocess==0.7.0
    pandas==2.0.3
    pytest==8.1.1
    ipython==8.12.3
    psutil==5.9.8
    python-dateutil==2.9.0.post0
    tzdata==2024.1
    Jinja2==3.0.3
    PyYAML==6.0.1
    iniconfig==2.0.0
    backcall==0.2.0
    bitstring==4.1.4
    distro==1.9.0
    contourpy==1.1.1
    opencv-python==4.9.0.80
    seaborn==0.13.2
    pillow==10.2.0
    kiwisolver==1.4.5
    rospkg==1.5.0
    gnupg==2.3.1
    pymlg @ git+https://github.com/decargroup/pymlg@main
    pyuwbcalib @ git+https://github.com/decargroup/uwb_calibration@master
    catkin-pkg==1.0.0
    exceptiongroup==1.2.0
    pure-eval==0.2.2
    matplotlib==3.7.5
    fonttools==4.50.0
    bagpy==0.5
    bitarray==2.9.2
    pyparsing==3.1.2
    pytz==2024.1
    pexpect==4.9.0
    Pygments==2.17.2
    docutils==0.20.1
    decorator==5.1.1
    pluggy==1.4.0
    -e git+ssh://git@github.com/decargroup/miluv.git@main#egg=miluv
    tomli==2.0.1
    six==1.16.0
    prompt-toolkit==3.0.43
    asttokens==2.4.1
    importlib_resources==6.4.0
    wcwidth==0.2.13
    ```
*   **setup.py**:
    ```python
    import setuptools

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name="miluv",
        version="1.0.0",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="<>",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.8",
        install_requires=[
            "pyuwbcalib @ git+https://github.com/decargroup/uwb_calibration@master",
            "pymlg @ git+https://github.com/decargroup/pymlg@main",
            "opencv-python>=4.9.0.80"
        ],
    )
    ```
*   **config/**
    *   **experiments.csv**:
        ```csv
        experiment,num_robots,num_tags_per_robot,num_anchors,anchor_constellation,trajectory,cir_bool,obstacles_bool,apriltags_bool,barometer_bool
        default_3_movingTriangle_0b,3,2,6,0b,moving_triangle,false,false,true,true
        default_3_zigzag_0,3,2,6,0,zigzag,false,false,true,false
        default_3_zigzag_1,3,2,6,1,zigzag,false,false,true,true
        default_3_zigzag_2,3,2,6,2,zigzag,false,false,true,true
        default_3_random_0,3,2,6,0,random,false,false,true,false
        default_3_random_0b,3,2,6,0b,random,false,false,true,true
        default_3_random2_0,3,2,6,0,random2,false,false,true,false
        default_3_random3_0b,3,2,6,0b,random3,false,false,true,true
        default_3_random3_1,3,2,6,1,random3,false,false,true,true
        ```
*   **examples/**
    *   **detect_apriltags.py**:
        ```python
        # Code retrieved from: https://pyimagesearch.com/2020/11/02/apriltag-with-python/
        # Authored by Dr. Adrian Rosebrock on November 2nd, 2020

        # Code modified by Nicholas Dahdah

        from miluv.data import DataLoader
        import os
        import cv2

        # To run this example, you need to install the following extra packages (found in requirements_dev.txt):
        import apriltag


        def main():
            mv = DataLoader(
                "default_3_random_0",
                cir=False,
                barometer=False,
            )

            img_path = os.path.join(
                mv.exp_dir,
                mv.exp_name,
                "ifo002",
                "color",
            )

            imgs = [
                cv2.imread(os.path.join(img_path, img)) for img in os.listdir(img_path)
            ]

            # YOUR APRILTAG DETECTION CODE BELOW
            gray_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
            options = apriltag.DetectorOptions(families="tag36h11")
            detector = apriltag.Detector(options)
            results = [detector.detect(gray) for gray in gray_imgs]

            # THIS IS WHERE YOU WOULD PROCESS THE APRILTAG DETECTION RESULTS
            for image, result in zip(imgs, results):
                for r in result:
                    # extract the bounding box (x, y)-coordinates for the AprilTag
                    # and convert each of the (x, y)-coordinate pairs to integers
                    (ptA, ptB, ptC, ptD) = r.corners
                    ptB = (int(ptB[0]), int(ptB[1]))
                    ptC = (int(ptC[0]), int(ptC[1]))
                    ptD = (int(ptD[0]), int(ptD[1]))
                    ptA = (int(ptA[0]), int(ptA[1]))
                    # draw the bounding box of the AprilTag detection
                    cv2.line(image, ptA, ptB, (0, 255, 0), 2)
                    cv2.line(image, ptB, ptC, (0, 255, 0), 2)
                    cv2.line(image, ptC, ptD, (0, 255, 0), 2)
                    cv2.line(image, ptD, ptA, (0, 255, 0), 2)
                    # draw the center (x, y)-coordinates of the AprilTag
                    (cX, cY) = (int(r.center[0]), int(r.center[1]))
                    cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)
                    # draw the tag family on the image
                    tagFamily = r.tag_family.decode("utf-8")
                    cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # show the output image after AprilTag detection
                cv2.imshow("Image", image)
                key = cv2.waitKey(0)

                # press the escape key to break from the loop
                if key == 27:
                    break

            cv2.destroyAllWindows()


        if __name__ == "__main__":
            main()
        ```
    *   **ekf_imu_one_robot.py**:
        ```python
        # %%
        import numpy as np
        import pandas as pd

        import sys

        from miluv.data import DataLoader
        import miluv.utils as utils
        import examples.ekfutils.imu_one_robot_models as model
        import examples.ekfutils.common as common

        def run_ekf_imu_one_robot(exp_name: str):
            
            #################### LOAD SENSOR DATA ####################
            miluv = DataLoader(exp_name, imu = "px4", cam = None, mag = False)
            data = miluv.data["ifo001"]

            #################### ALIGN SENSOR DATA TIMESTAMPS ####################
            # Set the query timestamps to be all the timestamps where UWB range or height data is available
            query_timestamps = np.append(
                data["uwb_range"]["timestamp"].to_numpy(), data["height"]["timestamp"].to_numpy()
            )
            query_timestamps = np.sort(np.unique(query_timestamps))

            imu_at_query_timestamps = miluv.query_by_timestamps(query_timestamps, robots="ifo001", sensors="imu_px4")["ifo001"]
            accel: pd.DataFrame = imu_at_query_timestamps["imu_px4"][["timestamp", "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"]]
            gyro: pd.DataFrame = imu_at_query_timestamps["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]

            #################### LOAD GROUND TRUTH DATA ####################
            gt_se23 = utils.get_se23_poses(
                data["mocap_quat"](query_timestamps), data["mocap_pos"].derivative(nu=1)(query_timestamps), data["mocap_pos"](query_timestamps)
            )
            gt_bias = imu_at_query_timestamps["imu_px4"][[
                "gyro_bias.x", "gyro_bias.y", "gyro_bias.z", 
                "accel_bias.x", "accel_bias.y", "accel_bias.z"
            ]].to_numpy()

            #################### EKF ####################
            # Initialize a variable to store the EKF state and covariance at each query timestamp for postprocessing
            ekf_history = {
                "pose": common.MatrixStateHistory(state_dim=5, covariance_dim=9),
                "bias": common.VectorStateHistory(state_dim=6)
            }

            # Initialize the EKF with the first ground truth pose, the anchor positions, and UWB tag moment arms
            ekf = model.EKF(gt_se23[0], miluv.anchors, miluv.tag_moment_arms)

            # Iterate through the query timestamps
            for i in range(0, len(query_timestamps)):
                # Get the gyro and vins data at this query timestamp for the EKF input
                input = np.array([
                    gyro.iloc[i]["angular_velocity.x"], gyro.iloc[i]["angular_velocity.y"], 
                    gyro.iloc[i]["angular_velocity.z"], accel.iloc[i]["linear_acceleration.x"], 
                    accel.iloc[i]["linear_acceleration.y"], accel.iloc[i]["linear_acceleration.z"]
                ])
                
                # Do an EKF prediction using the gyro and vins data
                dt = (query_timestamps[i] - query_timestamps[i - 1]) if i > 0 else 0
                ekf.predict(input, dt)
                
                # Check if range data is available at this query timestamp, and do an EKF correction
                range_idx = np.where(data["uwb_range"]["timestamp"] == query_timestamps[i])[0]
                if len(range_idx) > 0:
                    range_data = data["uwb_range"].iloc[range_idx]
                    ekf.correct({
                        "range": float(range_data["range"].iloc[0]),
                        "to_id": int(range_data["to_id"].iloc[0]),
                        "from_id": int(range_data["from_id"].iloc[0])
                    })
                    
                # Check if height data is available at this query timestamp, and do an EKF correction
                height_idx = np.where(data["height"]["timestamp"] == query_timestamps[i])[0]
                if len(height_idx) > 0:
                    height_data = data["height"].iloc[height_idx]
                    ekf.correct({"height": float(height_data["range"].iloc[0])})
                    
                # Store the EKF state and covariance at this query timestamp
                ekf_history["pose"].add(query_timestamps[i], ekf.pose, ekf.pose_covariance)
                ekf_history["bias"].add(query_timestamps[i], ekf.bias, ekf.bias_covariance)

            #################### POSTPROCESS ####################
            analysis = model.EvaluateEKF(gt_se23, gt_bias, ekf_history, exp_name)

            analysis.plot_error()
            analysis.plot_poses()
            analysis.plot_bias_error()
            analysis.save_results()

        if __name__ == "__main__":
            if len(sys.argv) < 2:
                exp_name = "default_1_random3_0"
            else:
                exp_name = sys.argv[1]
            
            run_ekf_imu_one_robot(exp_name)
        ```
    *   **ekf_imu_three_robots.py**:
        ```python
        # %%
        import numpy as np
        import pandas as pd

        import sys

        from miluv.data import DataLoader
        import miluv.utils as utils
        import examples.ekfutils.imu_three_robots_models as model
        import examples.ekfutils.common as common

        def run_ekf_imu_three_robots(exp_name: str):
            #################### LOAD SENSOR DATA ####################
            miluv = DataLoader(exp_name, imu = "px4", cam = None, mag = False)
            data = miluv.data

            # Merge the UWB range and height data from all robots into a single dataframe
            uwb_range = pd.concat([data[robot]["uwb_range"] for robot in data.keys()])
            height = pd.concat([data[robot]["height"].assign(robot=robot) for robot in data.keys()])

            #################### ALIGN SENSOR DATA TIMESTAMPS ####################
            # Set the query timestamps to be all the timestamps where UWB range or height data is available
            query_timestamps = np.append(uwb_range["timestamp"].to_numpy(), height["timestamp"].to_numpy())
            query_timestamps = np.sort(np.unique(query_timestamps))

            imu_at_query_timestamps = {
                robot: miluv.query_by_timestamps(query_timestamps, robots=robot, sensors="imu_px4")[robot]
                for robot in data.keys()
            }
            gyro: pd.DataFrame = {
                robot: imu_at_query_timestamps[robot]["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
                for robot in data.keys()
            }
            accel: pd.DataFrame = {
                robot: imu_at_query_timestamps[robot]["imu_px4"][["timestamp", "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"]]
                for robot in data.keys()
            }

            #################### LOAD GROUND TRUTH DATA ####################
            gt_se23 = {
                robot: utils.get_se23_poses(
                    data[robot]["mocap_quat"](query_timestamps), data[robot]["mocap_pos"].derivative(nu=1)(query_timestamps), data[robot]["mocap_pos"](query_timestamps)
                )
                for robot in data.keys()
            }
            gt_bias = {
                robot: imu_at_query_timestamps[robot]["imu_px4"][[
                    "gyro_bias.x", "gyro_bias.y", "gyro_bias.z", 
                    "accel_bias.x", "accel_bias.y", "accel_bias.z"
                ]].to_numpy()
                for robot in data.keys()
            }

            #################### EKF ####################
            # Initialize a variable to store the EKF state and covariance at each query timestamp for postprocessing
            ekf_history = {
                robot: {
                    "pose": common.MatrixStateHistory(state_dim=5, covariance_dim=9),
                    "bias": common.VectorStateHistory(state_dim=6)
                }
                for robot in data.keys()
            }

            # Initialize the EKF with the first ground truth pose, the anchor positions, and UWB tag moment arms
            ekf = model.EKF(
                {robot: gt_se23[robot][0] for robot in data.keys()}, 
                miluv.anchors, 
                miluv.tag_moment_arms
            )

            # Iterate through the query timestamps
            for i in range(0, len(query_timestamps)):
                # Get the gyro and vins data at this query timestamp for the EKF input
                input = {
                    robot: np.array([
                        gyro[robot].iloc[i]["angular_velocity.x"], gyro[robot].iloc[i]["angular_velocity.y"], 
                        gyro[robot].iloc[i]["angular_velocity.z"], accel[robot].iloc[i]["linear_acceleration.x"], 
                        accel[robot].iloc[i]["linear_acceleration.y"], accel[robot].iloc[i]["linear_acceleration.z"]
                    ])
                    for robot in data.keys()
                }
                
                # Do an EKF prediction using the gyro and vins data
                dt = (query_timestamps[i] - query_timestamps[i - 1]) if i > 0 else 0
                ekf.predict(input, dt)
                
                # Check if range data is available at this query timestamp, and do an EKF correction
                range_idx = np.where(uwb_range["timestamp"] == query_timestamps[i])[0]
                if len(range_idx) > 0:
                    range_data = uwb_range.iloc[range_idx]
                    ekf.correct({
                        "range": float(range_data["range"].iloc[0]),
                        "to_id": int(range_data["to_id"].iloc[0]),
                        "from_id": int(range_data["from_id"].iloc[0])
                    })
                    
                # Check if height data is available at this query timestamp, and do an EKF correction
                height_idx = np.where(height["timestamp"] == query_timestamps[i])[0]
                if len(height_idx) > 0:
                    height_data = height.iloc[height_idx]
                    ekf.correct({
                        "height": float(height_data["range"].iloc[0]),
                        "robot": height_data["robot"].iloc[0]
                    })
                    
                # Store the EKF state and covariance at this query timestamp
                for robot in data.keys():
                    ekf_history[robot]["pose"].add(query_timestamps[i], ekf.pose[robot], ekf.pose_covariance[robot])
                    ekf_history[robot]["bias"].add(query_timestamps[i], ekf.bias[robot], ekf.bias_covariance[robot])

            #################### POSTPROCESS ####################
            analysis = model.EvaluateEKF(gt_se23, gt_bias, ekf_history, exp_name)

            analysis.plot_error()
            analysis.plot_poses()
            analysis.plot_bias_error()
            analysis.save_results()

        if __name__ == "__main__":
            if len(sys.argv) < 2:
                exp_name = "default_3_random_0"
            else:
                exp_name = sys.argv[1]
            
            run_ekf_imu_three_robots(exp_name)
        ```
    *   **ekf_vins_one_robot.py**:
        ```python
        # %%
        import numpy as np
        import pandas as pd

        import sys

        from miluv.data import DataLoader
        import miluv.utils as utils
        import examples.ekfutils.vins_one_robot_models as model
        import examples.ekfutils.common as common

        def run_ekf_vins_one_robot(exp_name: str):
            #################### LOAD SENSOR DATA ####################
            miluv = DataLoader(exp_name, imu = "px4", cam = None, mag = False, remove_imu_bias = True)
            data = miluv.data["ifo001"]
            vins = utils.load_vins(exp_name, "ifo001", loop = False, postprocessed = True)

            #################### ALIGN SENSOR DATA TIMESTAMPS ####################
            # Set the query timestamps to be all the timestamps where UWB range or height data is available
            # and within the time range of the VINS data
            query_timestamps = np.append(
                data["uwb_range"]["timestamp"].to_numpy(), data["height"]["timestamp"].to_numpy()
            )
            query_timestamps = query_timestamps[query_timestamps > vins["timestamp"].iloc[0]]
            query_timestamps = query_timestamps[query_timestamps < vins["timestamp"].iloc[-1]]
            query_timestamps = np.sort(np.unique(query_timestamps))

            imu_at_query_timestamps = miluv.query_by_timestamps(query_timestamps, robots="ifo001", sensors="imu_px4")["ifo001"]
            gyro: pd.DataFrame = imu_at_query_timestamps["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
            vins_at_query_timestamps = utils.zero_order_hold(query_timestamps, vins)

            #################### LOAD GROUND TRUTH DATA ####################
            gt_se3 = utils.get_se3_poses(
                data["mocap_quat"](query_timestamps), data["mocap_pos"](query_timestamps)
            )

            # Use ground truth data to convert VINS data from the absolute (mocap) frame to the robot's body frame
            vins_body_frame = common.convert_vins_velocity_to_body_frame(vins_at_query_timestamps, gt_se3)

            #################### EKF ####################
            # Initialize a variable to store the EKF state and covariance at each query timestamp for postprocessing
            ekf_history = common.MatrixStateHistory(state_dim=4, covariance_dim=6)

            # Initialize the EKF with the first ground truth pose, the anchor postions, and UWB tag moment arms
            ekf = model.EKF(gt_se3[0], miluv.anchors, miluv.tag_moment_arms)

            # Iterate through the query timestamps
            for i in range(0, len(query_timestamps)):
                # Get the gyro and vins data at this query timestamp for the EKF input
                input = np.array([
                    gyro.iloc[i]["angular_velocity.x"], gyro.iloc[i]["angular_velocity.y"], 
                    gyro.iloc[i]["angular_velocity.z"], vins_body_frame.iloc[i]["twist.linear.x"],
                    vins_body_frame.iloc[i]["twist.linear.y"], vins_body_frame.iloc[i]["twist.linear.z"],
                ])
                
                # Do an EKF prediction using the gyro and vins data
                dt = (query_timestamps[i] - query_timestamps[i - 1]) if i > 0 else 0
                ekf.predict(input, dt)
                
                # Check if range data is available at this query timestamp, and do an EKF correction
                range_idx = np.where(data["uwb_range"]["timestamp"] == query_timestamps[i])[0]
                if len(range_idx) > 0:
                    range_data = data["uwb_range"].iloc[range_idx]
                    ekf.correct({
                        "range": float(range_data["range"].iloc[0]),
                        "to_id": int(range_data["to_id"].iloc[0]),
                        "from_id": int(range_data["from_id"].iloc[0])
                    })
                    
                # Check if height data is available at this query timestamp, and do an EKF correction
                height_idx = np.where(data["height"]["timestamp"] == query_timestamps[i])[0]
                if len(height_idx) > 0:
                    height_data = data["height"].iloc[height_idx]
                    ekf.correct({"height": float(height_data["range"].iloc[0])})
                    
                # Store the EKF state and covariance at this query timestamp
                ekf_history.add(query_timestamps[i], ekf.x, ekf.P)

            #################### POSTPROCESS ####################
            analysis = model.EvaluateEKF(gt_se3, ekf_history, exp_name)

            analysis.plot_error()
            analysis.plot_poses()
            analysis.save_results()

        if __name__ == "__main__":
            if len(sys.argv) < 2:
                exp_name = "default_1_random3_0"
            else:
                exp_name = sys.argv[1]
            
            run_ekf_vins_one_robot(exp_name)
        ```
    *   **ekf_vins_three_robots.py**:
        ```python
        # %%
        from miluv.data import DataLoader
        import miluv.utils as utils
        import examples.ekfutils.vins_three_robots_models as model
        import examples.ekfutils.common as common

        import sys

        import numpy as np
        import pandas as pd

        def run_ekf_vins_three_robots(exp_name: str):
            #################### LOAD SENSOR DATA ####################
            miluv = DataLoader(exp_name, imu = "px4", cam = None, mag = False, remove_imu_bias = True)
            data = miluv.data
            vins = {robot: utils.load_vins(exp_name, robot, loop = False, postprocessed = True) for robot in data.keys()}

            # Merge the UWB range and height data from all robots into a single dataframe
            uwb_range = pd.concat([data[robot]["uwb_range"] for robot in data.keys()])
            height = pd.concat([data[robot]["height"].assign(robot=robot) for robot in data.keys()])

            #################### ALIGN SENSOR DATA TIMESTAMPS ####################
            # Set the query timestamps to be all the timestamps where UWB range or height data is available
            # and within the time range of the VINS data
            query_timestamps = np.append(uwb_range["timestamp"].to_numpy(), height["timestamp"].to_numpy())
            query_timestamps = query_timestamps[
                (query_timestamps > vins["ifo001"]["timestamp"].iloc[0]) & (query_timestamps < vins["ifo001"]["timestamp"].iloc[-1]) &
                (query_timestamps > vins["ifo002"]["timestamp"].iloc[0]) & (query_timestamps < vins["ifo002"]["timestamp"].iloc[-1]) &
                (query_timestamps > vins["ifo003"]["timestamp"].iloc[0]) & (query_timestamps < vins["ifo003"]["timestamp"].iloc[-1])
            ]
            query_timestamps = np.sort(np.unique(query_timestamps))

            imu_at_query_timestamps = {
                robot: miluv.query_by_timestamps(query_timestamps, robots=robot, sensors="imu_px4")[robot]
                for robot in data.keys()
            }
            gyro: pd.DataFrame = {
                robot: imu_at_query_timestamps[robot]["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]]
                for robot in data.keys()
            }
            vins_at_query_timestamps = {
                robot: utils.zero_order_hold(query_timestamps, vins[robot]) for robot in data.keys()
            }

            #################### LOAD GROUND TRUTH DATA ####################
            gt_se3 = {
                robot: utils.get_se3_poses(data[robot]["mocap_quat"](query_timestamps), data[robot]["mocap_pos"](query_timestamps)) 
                for robot in data.keys()
            }

            # Use ground truth data to convert VINS data from the absolute (mocap) frame to the robot's body frame
            vins_body_frame = {
                robot: common.convert_vins_velocity_to_body_frame(vins_at_query_timestamps[robot], gt_se3[robot]) 
                for robot in data.keys()
            }

            #################### EKF ####################
            # Initialize a variable to store the EKF state and covariance at each query timestamp for postprocessing
            ekf_history = {
                robot: common.MatrixStateHistory(state_dim=4, covariance_dim=6) for robot in data.keys()
            }

            # Initialize the EKF with the first ground truth pose, the anchor postions, and UWB tag moment arms
            ekf = model.EKF(
                {robot: gt_se3[robot][0] for robot in data.keys()}, 
                miluv.anchors, 
                miluv.tag_moment_arms
            )

            # Iterate through the query timestamps
            for i in range(0, len(query_timestamps)):
                # Get the gyro and vins data at this query timestamp for the EKF input
                input = {
                    robot: np.array([
                        gyro[robot].iloc[i]["angular_velocity.x"], gyro[robot].iloc[i]["angular_velocity.y"], 
                        gyro[robot].iloc[i]["angular_velocity.z"], vins_body_frame[robot].iloc[i]["twist.linear.x"],
                        vins_body_frame[robot].iloc[i]["twist.linear.y"], vins_body_frame[robot].iloc[i]["twist.linear.z"],
                    ])
                    for robot in data.keys()
                }
                
                # Do an EKF prediction using the gyro and vins data
                dt = (query_timestamps[i] - query_timestamps[i - 1]) if i > 0 else 0
                ekf.predict(input, dt)
                
                # Check if range data is available at this query timestamp, and do an EKF correction
                range_idx = np.where(uwb_range["timestamp"] == query_timestamps[i])[0]
                if len(range_idx) > 0:
                    range_data = uwb_range.iloc[range_idx]
                    ekf.correct({
                        "range": float(range_data["range"].iloc[0]),
                        "to_id": int(range_data["to_id"].iloc[0]),
                        "from_id": int(range_data["from_id"].iloc[0])
                    })
                    
                # Check if height data is available at this query timestamp, and do an EKF correction
                height_idx = np.where(height["timestamp"] == query_timestamps[i])[0]
                if len(height_idx) > 0:
                    height_data = height.iloc[height_idx]
                    ekf.correct({
                        "height": float(height_data["range"].iloc[0]),
                        "robot": height_data["robot"].iloc[0]
                    })
                    
                # Store the EKF state and covariance at this query timestamp
                for robot in data.keys():
                    ekf_history[robot].add(query_timestamps[i], ekf.x[robot], ekf.get_covariance(robot))

            #################### POSTPROCESS ####################
            analysis = model.EvaluateEKF(gt_se3, ekf_history, exp_name)

            analysis.plot_error()
            analysis.plot_poses()
            analysis.save_results()
            
        if __name__ == "__main__":
            if len(sys.argv) < 2:
                exp_name = "default_3_random_0"
            else:
                exp_name = sys.argv[1]
            
            run_ekf_vins_three_robots(exp_name)
        ```
    *   **extract_data.py**:
        ```python
        # %%
        from miluv.data import DataLoader

        mv = DataLoader("default_3_random_0")
        # %%
        ```
    *   **los_classification.py**:
        ```python
        import numpy as np
        from miluv.data import DataLoader

        # To run this example, you need to install the following extra packages (found in requirements_dev.txt):
        from sklearn.model_selection import train_test_split
        from lazypredict.Supervised import LazyClassifier


        def main():
            # List of anchor IDs that are NLOS
            tag_ids_nlos = [
                1,  # styrofoam
                3,  # plastic
                4,  # wood
            ]

            mv = DataLoader(
                "cirObstacles_3_random_0",
                barometer=False,
                cir=True
            )

            X = []
            y = []

            for robot_id in mv.data.keys():
                for anchor_id, cir_data in zip(
                        mv.data[robot_id]["uwb_cir"]["to_id"],
                        mv.data[robot_id]["uwb_cir"]["cir"],
                ):
                    cir_data = cir_data.replace("[", "").replace("]", "").split(", ")
                    cir_data = [int(x) for x in cir_data]

                    X.append(cir_data)
                    if anchor_id in tag_ids_nlos:
                        y.append(1)
                    else:
                        y.append(0)

            X = np.array(X)
            y = np.array(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=0,
            )

            # YOUR ML CLASSIFIER GOES HERE
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)
            print(models)


        if __name__ == "__main__":
            main()
        ```
    *   **run_vins.sh**:
        ```bash
        #!/bin/bash

        # Check if both arguments are provided
        if [ $# -ne 2 ]; then
            echo "Usage: $0 <exp_name> <robot>"
            exit 1
        fi

        # Assign the arguments to variables
        exp_name=$1
        robot=$2

        if [ ! -d "$(rospack find miluv)/data/vins" ]; then
            mkdir -p $(rospack find miluv)/data/vins
        fi

        # Get the duration of the bag file in seconds
        bag=$(rospack find miluv)/data/${exp_name}/${robot}.bag
        bag_duration=$(rosbag info --yaml "$bag" | grep "duration:" | awk '{print $2}' | cut -d'.' -f1)
        echo "Bag duration is ${bag_duration}s."

        # Run the VINS node
        timeout $bag_duration roslaunch miluv vins.launch robot:=$robot bag:=$bag > /dev/null 2>&1

        cd $(rospack find miluv)/data/vins
        mkdir -p $exp_name
        mv vio.csv $exp_name/${robot}_vio.csv
        mv vio_loop.csv $exp_name/${robot}_vio_loop.csv
        ```
    *   **vins.Dockerfile**:
        ```dockerfile
        # Use the official ROS Melodic image as the base image
        FROM ros:melodic

        # Set environment variables
        ENV LANG C.UTF-8
        ENV LC_ALL C.UTF-8

        # Set working directory
        WORKDIR /workspace

        # Install ROS Melodic and dependencies
        RUN apt-get update && apt-get install -y \
            software-properties-common \
            curl \
            gnupg2 \
            lsb-release \
            wget \
            unzip \
            build-essential \
            git \
            cmake \
            libopencv-dev \
            libeigen3-dev \
            libboost-all-dev \
            libgflags-dev \
            libgtest-dev \
            libyaml-cpp-dev \
            libsuitesparse-dev \
            libgoogle-glog-dev \
            python3-pip \
            python3-colcon-common-extensions \
            ros-melodic-vision-opencv \
            ros-melodic-pcl-ros \
            ros-melodic-tf2 \
            ros-melodic-tf2-ros \
            ros-melodic-std-msgs \
            ros-melodic-geometry-msgs \
            ros-melodic-sensor-msgs \
            ros-melodic-image-transport \
            ros-melodic-compressed-image-transport \
            ros-melodic-rviz \
            && rm -rf /var/lib/apt/lists/*

        # Install Python 3.10 and dependencies
        RUN apt-get update && apt-get install -y \
            software-properties-common \
            wget \
            build-essential \
            libssl-dev \
            zlib1g-dev \
            libncurses5-dev \
            libnss3-dev \
            libsqlite3-dev \
            libreadline-dev \
            libffi-dev \
            curl \
            && rm -rf /var/lib/apt/lists/*

        # Download and build Python 3.10
        RUN wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz \
            && tar -xvf Python-3.10.12.tgz \
            && cd Python-3.10.12 \
            && ./configure --enable-optimizations \
            && make \
            && make install \
            && cd .. \
            && rm -rf Python-3.10.12.tgz Python-3.10.12

        # Update alternatives to make python3 point to python3.10
        RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1
        RUN update-alternatives --set python3 /usr/local/bin/python3.10
        RUN python3 -m pip install --upgrade pip setuptools wheel
        RUN python3 -m pip install numpy pandas csaps scipy matplotlib

        # Install Ceres Solver and additional dependencies
        RUN sudo ln -s /usr/share/pyshared/lsb_release.py /usr/local/lib/python3.10/site-packages/lsb_release.py
        RUN apt-get update && apt-get install -y \
            libceres-dev \
            libatlas-base-dev \
            && rm -rf /var/lib/apt/lists/*

        # Create a workspace and clone the VINS-Fusion repository and the MILUV repository
        RUN mkdir -p /workspace/src
        WORKDIR /workspace/src
        RUN git clone https://github.com/decargroup/miluv.git
        RUN git clone https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git

        # Make symlink to build uwb_ros with UWB messages
        RUN ln -s miluv/uwb_ros .

        # Install MILUV 
        WORKDIR /workspace/src/miluv
        RUN /bin/bash -c "pip3 install ."

        # Build the workspace
        WORKDIR /workspace
        RUN /bin/bash -c "source /opt/ros/melodic/setup.bash && catkin_make -DCMAKE_SUPPRESS_DEVELOPER_WARNINGS=ON"

        # Source ROS setup
        RUN echo "source /workspace/devel/setup.bash" >> ~/.bashrc

        # Expose the necessary ports
        EXPOSE 11311

        # Set entrypoint to bash so you can run commands interactively
        WORKDIR /workspace/src/miluv
        CMD ["/bin/bash"]
        ```
    *   **vins_docker_compose.yaml**:
        ```yaml
        version: '3.9'

        services:
          miluv:
            image: miluv-vins
            runtime: nvidia
            privileged: true
            network_mode: host
            environment:
              DISPLAY: $DISPLAY
              NVIDIA_VISIBLE_DEVICES: all
              NVIDIA_DRIVER_CAPABILITIES: all
            volumes:
              - ../data:/workspace/src/miluv/data:rw
              - /tmp/.X11-unix:/tmp/.X11-unix:rw
            deploy:
              resources:
                reservations:
                  devices:
                    - driver: nvidia
                      capabilities: [gpu]
            stdin_open: true
            tty: true
        ```
    *   **visualize_imu.py**:
        ```python
        from miluv.data import DataLoader
        import miluv.utils as utils

        import matplotlib
        matplotlib.use('Qt5Agg')

        import matplotlib.pyplot as plt

        plt.rcParams['axes.grid'] = True

        #################### EXPERIMENT DETAILS ####################
        exp_name = "default_3_random_0"
        robot = "ifo001"

        #################### LOAD SENSOR DATA ####################
        miluv = DataLoader(exp_name, cam = None, mag = False)
        data = miluv.data[robot]

        imu_px4 = data["imu_px4"]
        time = imu_px4["timestamp"]
        pos = data["mocap_pos"](time)
        quat = data["mocap_quat"](time)

        #################### GROUND TRUTH IMU ####################
        gt_gyro = utils.get_angular_velocity_splines(time, data["mocap_quat"])(time)
        gt_accelerometer = utils.get_accelerometer_splines(time, data["mocap_pos"], data["mocap_quat"])(time)

        #################### VISUALIZE GYROSCOPE ####################
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("Gyroscope")

        axs[0].plot(time, imu_px4["angular_velocity.x"], label="IMU PX4 Measurement")
        axs[1].plot(time, imu_px4["angular_velocity.y"], label="IMU PX4 Measurement")
        axs[2].plot(time, imu_px4["angular_velocity.z"], label="IMU PX4 Measurement")

        axs[0].plot(time, gt_gyro[0, :], label="Ground Truth")
        axs[1].plot(time, gt_gyro[1, :], label="Ground Truth")
        axs[2].plot(time, gt_gyro[2, :], label="Ground Truth")

        axs[0].set_ylabel("Gyro X (rad/s)")
        axs[1].set_ylabel("Gyro Y (rad/s)")
        axs[2].set_ylabel("Gyro Z (rad/s)")

        axs[0].set_ylim([-1, 1])
        axs[1].set_ylim([-1, 1])
        axs[2].set_ylim([-1, 1])

        axs[0].legend()

        #################### VISUALIZE GYROSCOPE ERROR AND BIAS ####################
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("Gyroscope Error and Bias")

        axs[0].plot(time, gt_gyro[0, :] - imu_px4["angular_velocity.x"], label="Measurement Error")
        axs[1].plot(time, gt_gyro[1, :] - imu_px4["angular_velocity.y"], label="Measurement Error")
        axs[2].plot(time, gt_gyro[2, :] - imu_px4["angular_velocity.z"], label="Measurement Error")

        axs[0].plot(time, imu_px4["gyro_bias.x"], label="IMU Bias")
        axs[1].plot(time, imu_px4["gyro_bias.y"], label="IMU Bias")
        axs[2].plot(time, imu_px4["gyro_bias.z"], label="IMU Bias")

        axs[0].set_ylabel("Gyro X (rad/s)")
        axs[1].set_ylabel("Gyro Y (rad/s)")
        axs[2].set_ylabel("Gyro Z (rad/s)")

        axs[0].set_ylim([-0.5, 0.5])
        axs[1].set_ylim([-0.5, 0.5])
        axs[2].set_ylim([-0.5, 0.5])

        axs[0].legend()

        #################### VISUALIZE ACCELEROMETER ####################
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("Accelerometer")

        axs[0].plot(time, imu_px4["linear_acceleration.x"], label="IMU PX4 Measurement")
        axs[1].plot(time, imu_px4["linear_acceleration.y"], label="IMU PX4 Measurement")
        axs[2].plot(time, imu_px4["linear_acceleration.z"], label="IMU PX4 Measurement")

        axs[0].plot(time, gt_accelerometer[0, :], label="Ground Truth")
        axs[1].plot(time, gt_accelerometer[1, :], label="Ground Truth")
        axs[2].plot(time, gt_accelerometer[2, :], label="Ground Truth")

        axs[0].set_ylabel("Accel X (m/s^2)")
        axs[1].set_ylabel("Accel Y (m/s^2)")
        axs[2].set_ylabel("Accel Z (m/s^2)")

        axs[0].set_ylim([-5, 5])
        axs[1].set_ylim([-5, 5])
        axs[2].set_ylim([5, 15])

        axs[0].legend()

        #################### VISUALIZE ACCELEROMETER ERROR AND BIAS ####################
        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("Accelerometer Error and Bias")

        axs[0].plot(time, gt_accelerometer[0, :] - imu_px4["linear_acceleration.x"], label="Measurement Error")
        axs[1].plot(time, gt_accelerometer[1, :] - imu_px4["linear_acceleration.y"], label="Measurement Error")
        axs[2].plot(time, gt_accelerometer[2, :] - imu_px4["linear_acceleration.z"], label="Measurement Error")

        axs[0].plot(time, imu_px4["accel_bias.x"], label="IMU Bias")
        axs[1].plot(time, imu_px4["accel_bias.y"], label="IMU Bias")
        axs[2].plot(time, imu_px4["accel_bias.z"], label="IMU Bias")

        axs[0].set_ylabel("Accel X (m/s^2)")
        axs[1].set_ylabel("Accel Y (m/s^2)")
        axs[2].set_ylabel("Accel Z (m/s^2)")

        axs[0].set_ylim([-3, 3])
        axs[1].set_ylim([-3, 3])
        axs[2].set_ylim([-3, 3])

        axs[0].legend()

        plt.show(block=True)
        ```
    *   **__init__.py**:
        ```python
        ```
*   **launch/**
    *   **vins.launch**:
        ```xml
        <launch>
            <!-- Global parameters -->
            <arg name="robot"/>
            <arg name="bag"/>
            <arg name="offline" default="true"/>
            <arg name="use_rviz" default="true"/>
            <arg name="do_loop_closures" default="true"/>

            <!-- Launch RVIZ -->
            <group if="$(arg use_rviz)">
                <include file="$(find vins)/launch/vins_rviz.launch">
                </include>
            </group>

            <!-- Launch VINS -->
            <node name="vins_estimator" output="screen" pkg="vins" type="vins_node"
                args="$(find miluv)/config/vins/$(arg robot)/vins.yaml"/>
            
            <!-- Launch loop closures -->
            <group if="$(arg do_loop_closures)">
                <node name="loop_fusion" output="screen" pkg="loop_fusion" type="loop_fusion_node"
                args="$(find miluv)/config/vins/$(arg robot)/vins.yaml"/>
            </group>

            <!-- Run image decompressor -->
            <group if="$(arg offline)">
                <node name="infra1_decompressor" output="screen" pkg="image_transport" type="republish"
                    args="compressed in:=/$(arg robot)/camera/infra1/image_rect_raw raw out:=/$(arg robot)/camera/infra1/image_raw"/>
                <node name="infra2_decompressor" output="screen" pkg="image_transport" type="republish"
                    args="compressed in:=/$(arg robot)/camera/infra2/image_rect_raw raw out:=/$(arg robot)/camera/infra2/image_raw"/>
            </group>

            <!-- Run bag file -->
            <group if="$(arg offline)">
                <node name="bag_player" output="screen" pkg="rosbag" type="play"
                    args="$(arg bag) -s 5"/>
            </group>

        </launch>
        ```
*   **miluv/**
    *   **data.py**:
        ```python
        import miluv.utils as utils

        import pandas as pd
        import numpy as np
        import cv2
        import os
        from typing import List

        class DataLoader:

            def __init__(
                self,
                exp_name: str,
                exp_dir: str = "./data",
                imu: str = "both",
                cam: list = [
                    "color",
                    "bottom",
                    "infra1",
                    "infra2",
                ],
                uwb: bool = True,
                height: bool = True,
                mag: bool = True,
                cir: bool = False,
                barometer: bool = False,
                remove_imu_bias: bool = False,
            ):

                # TODO: Add checks for valid exp dir and name
                self.exp_name = exp_name
                self.exp_dir = exp_dir
                self.cam = cam

                exp_data = pd.read_csv("config/experiments.csv")
                exp_data = exp_data[exp_data["experiment"].astype(str) == exp_name]
                
                robot_ids = [f"ifo00{i}" for i in range(1, exp_data["num_robots"].iloc[0] + 1)]
                self.anchors = utils.get_anchors()[exp_data["anchor_constellation"].iloc[0]]
                
                tag_moment_arms = utils.get_tag_moment_arms()
                self.tag_moment_arms = {id: tag_moment_arms[id] for id in robot_ids}
                
                self.data = {id: {} for id in robot_ids}
                for id in robot_ids:
                    mocap_df = self.read_csv("mocap", id)
                    self.data[id]["mocap_pos"], self.data[id]["mocap_quat"] \
                        = utils.get_mocap_splines(mocap_df)
                        
                    if imu == "both" or imu == "px4":
                        self.data[id].update({"imu_px4": []})
                        self.data[id]["imu_px4"] = self.read_csv("imu_px4", id)
                        
                        utils.add_imu_bias(
                            self.data[id]["imu_px4"], self.data[id]["mocap_pos"], self.data[id]["mocap_quat"]
                        )
                        
                        if remove_imu_bias:
                            self.data[id]["imu_px4"]["angular_velocity.x"] += self.data[id]["imu_px4"]["gyro_bias.x"]
                            self.data[id]["imu_px4"]["angular_velocity.y"] += self.data[id]["imu_px4"]["gyro_bias.y"]
                            self.data[id]["imu_px4"]["angular_velocity.z"] += self.data[id]["imu_px4"]["gyro_bias.z"]
                            
                            self.data[id]["imu_px4"]["linear_acceleration.x"] += self.data[id]["imu_px4"]["accel_bias.x"]
                            self.data[id]["imu_px4"]["linear_acceleration.y"] += self.data[id]["imu_px4"]["accel_bias.y"]
                            self.data[id]["imu_px4"]["linear_acceleration.z"] += self.data[id]["imu_px4"]["accel_bias.z"]
                            
                            self.data[id]["imu_px4"].drop(
                                columns=[
                                    "gyro_bias.x", "gyro_bias.y", "gyro_bias.z", 
                                    "accel_bias.x", "accel_bias.y", "accel_bias.z"
                                ], inplace=True
                            )
                        
                    if imu == "both" or imu == "cam":
                        self.data[id].update({"imu_cam": []})
                        self.data[id]["imu_cam"] = self.read_csv("imu_cam", id)
                        
                        utils.add_imu_bias(
                            self.data[id]["imu_cam"], self.data[id]["mocap_pos"], self.data[id]["mocap_quat"]
                        )
                        
                        if remove_imu_bias:
                            self.data[id]["imu_cam"]["angular_velocity.x"] += self.data[id]["imu_cam"]["gyro_bias.x"]
                            self.data[id]["imu_cam"]["angular_velocity.y"] += self.data[id]["imu_cam"]["gyro_bias.y"]
                            self.data[id]["imu_cam"]["angular_velocity.z"] += self.data[id]["imu_cam"]["gyro_bias.z"]
                            
                            self.data[id]["imu_cam"]["linear_acceleration.x"] += self.data[id]["imu_cam"]["accel_bias.x"]
                            self.data[id]["imu_cam"]["linear_acceleration.y"] += self.data[id]["imu_cam"]["accel_bias.y"]
                            self.data[id]["imu_cam"]["linear_acceleration.z"] += self.data[id]["imu_cam"]["accel_bias.z"]
                            
                            self.data[id]["imu_px4"].drop(
                                columns=[
                                    "gyro_bias.x", "gyro_bias.y", "gyro_bias.z", 
                                    "accel_bias.x", "accel_bias.y", "accel_bias.z"
                                ], inplace=True
                            )

                    if uwb:
                        self.data[id].update({"uwb_range": []})
                        self.data[id]["uwb_range"] = self.read_csv("uwb_range", id)

                        self.data[id].update({"uwb_passive": []})
                        self.data[id]["uwb_passive"] = self.read_csv("uwb_passive", id)

                    if cir:
                        self.data[id].update({"uwb_cir": []})
                        self.data[id]["uwb_cir"] = self.read_csv("uwb_cir", id)

                    if height:
                        self.data[id].update({"height": []})
                        self.data[id]["height"] = self.read_csv("height", id)
                        self.data[id]["height"]["range"] -= utils.get_height_bias(id)

                    if mag:
                        self.data[id].update({"mag": []})
                        self.data[id]["mag"] = self.read_csv("mag", id)

                    if barometer:
                        self.data[id].update({"barometer": []})
                        self.data[id]["barometer"] = self.read_csv("barometer", id)

                # TODO: Load timestamp-to-image mapping?
                # if cam == "both" or cam == "bottom":
                #     self.load_imgs("bottom")
                # if cam == "both" or cam == "front":
                #     self.load_imgs("front")

            def read_csv(self, topic: str, robot_id) -> pd.DataFrame:
                """Read a csv file for a given robot and topic."""
                path = os.path.join(self.exp_dir, self.exp_name, robot_id,
                                    topic + ".csv")
                df = pd.read_csv(path)
                
                df.drop_duplicates(subset="timestamp", inplace=True)
                df.sort_values(by="timestamp", inplace=True)
                
                return df

            def query_by_timestamps(
                self, 
                timestamps: np.ndarray, 
                robots: List = None, 
                sensors: List = None
            ) -> pd.DataFrame:
                """
                Get the data at one or more query times. The return data is at the lower bound 
                of the time window where data is available, i.e., a zero-order hold.

                Parameters
                ----------
                timestamps : np.ndarray
                    The query times for which data is requested.
                robots : List, optional
                    The robots for which data is requested. If None, data for all robots is returned.
                sensors : List, optional
                    The sensors for which data is requested. If None, data for all sensors is returned.

                Returns
                -------
                pd.DataFrame
                    The data at the query times.
                """
                timestamps = np.array(timestamps)

                if robots is None:
                    robots = self.data.keys()

                robots = [robots] if type(robots) is str else robots
                sensors = [sensors] if type(sensors) is str else sensors

                new_data: dict = {}
                for id in robots:
                    new_data[id] = {}
                    if sensors is None:
                        sensors = list(self.data[id].keys() - ["mocap"])

                    for sensor in sensors:
                        new_data[id][sensor] = utils.zero_order_hold(timestamps, self.data[id][sensor])

                return new_data

            def closest_past_timestamp(self, robot_id: str, sensor: str,
                                       timestamp: float) -> int:
                """Return the closest timestamp in the past for a given sensor."""
                not_over = None
                if sensor != "bottom" and sensor != "color" and sensor != "infra1" and sensor != "infra2":
                    not_over = [
                        ts for ts in self.data[robot_id][sensor]["timestamp"]
                        if ts <= timestamp
                    ]
                else:
                    all_imgs = os.listdir(
                        os.path.join(self.exp_dir, self.exp_name, robot_id, sensor))
                    all_imgs = [int(img.split(".")[0]) for img in all_imgs]
                    not_over = [ts for ts in all_imgs if ts <= timestamp]

                if not_over == []:
                    return None
                return max(not_over)

            def data_from_timestamp(
                self,
                timestamps: list,
                robot_ids=None,
                sensors=None,
            ) -> dict:
                """Return all data from a given timestamp."""

                def data_from_timestamp_robot(robot_id: str, timestamps: list) -> dict:
                    """Return all data from a given timestamp for a given robot."""
                    data_by_robot = {}
                    for sensor in sensors:
                        data_by_robot[sensor] = data_from_timestamp_sensor(
                            robot_id, sensor, timestamps)

                    return data_by_robot

                def data_from_timestamp_sensor(robot_id: str, sensor: str,
                                               timestamps: list) -> dict:
                    """Return all data from a given timestamp for a given sensor for a given robot."""
                    col_names = self.data[robot_id][sensor].columns
                    df = pd.DataFrame(columns=col_names)
                    for timestamp in timestamps:
                        if timestamp in self.data[robot_id][sensor][
                                "timestamp"].values:
                            df = pd.concat([
                                df if not df.empty else None, self.data[robot_id]
                                [sensor].loc[self.data[robot_id][sensor]["timestamp"]
                                             == timestamp]
                            ])
                        else:
                            df = pd.concat([
                                df if not df.empty else None, self.data[robot_id]
                                [sensor].loc[self.data[robot_id][sensor]["timestamp"]
                                             == self.closest_past_timestamp(
                                                 robot_id, sensor, timestamp)]
                            ])
                    return df

                if robot_ids is None:
                    robot_ids = self.data.keys()
                if sensors is None:
                    sensors = self.data['ifo001'].keys()

                data_by_timestamp = {}
                for robot_id in robot_ids:
                    data_by_timestamp[robot_id] = data_from_timestamp_robot(
                        robot_id, timestamps)

                return data_by_timestamp

            def imgs_from_timestamps(self,
                                     timestamps: list,
                                     robot_ids=None,
                                     cams=None) -> dict:
                """Return all images from a given timestamp."""

                def imgs_from_timestamp_robot(robot_id: str, cams: list,
                                              timestamps: list) -> dict:
                    """Return all images from a given timestamp for a given robot."""
                    img_by_robot = {}
                    for cam in cams:
                        valid_ts = []
                        imgs = []
                        for timestamp in timestamps:
                            if cam:
                                img_ts = self.closest_past_timestamp(
                                    robot_id, cam, timestamp)
                                if img_ts is None:
                                    # print("No", cam, "image found for timestamp",
                                    #       timestamp, "for robot_id", robot_id)    # Debugging msg
                                    continue
                                img_path = os.path.join(self.exp_dir, self.exp_name,
                                                        robot_id, cam,
                                                        str(img_ts) + ".jpeg")
                                imgs.append(cv2.imread(img_path))
                                valid_ts.append(img_ts)
                        img_by_robot[cam] = pd.DataFrame({
                            "timestamp": valid_ts,
                            "image": imgs
                        })
                    return img_by_robot

                if robot_ids is None:
                    robot_ids = self.data.keys()
                if cams is None:
                    cams = self.cam

                img_by_timestamp = {}
                for robot_id in robot_ids:
                    img_by_timestamp[robot_id] = imgs_from_timestamp_robot(
                        robot_id, cams, timestamps)
                return img_by_timestamp


        if __name__ == "__main__":
            mv = DataLoader(
                "1c",
                barometer=False,
                height=False,
            )

            print("done!")
        ```
    *   **utils.py**:
        ```python
        import numpy as np
        from csaps import csaps
        from csaps import ISmoothingSpline
        import pandas as pd
        from scipy.spatial.transform import Rotation
        import scipy as sp
        import yaml

        from pymlg import SO3, SE3, SE23

        def get_anchors() -> dict[str, dict[int, np.ndarray]]:
            """
            Get anchor positions.
            
            Returns: 
            dict
            - Anchor positions.
            """
            
            with open(f"config/uwb/anchors.yaml", "r") as file:
                anchor_positions = yaml.safe_load(file)
            
            for constellation in anchor_positions:
                anchors = list(anchor_positions[constellation].keys())
                for anchor in anchors:
                    pos = np.array([eval(anchor_positions[constellation][anchor])]).reshape(3, 1)
                    anchor_positions[constellation][int(anchor)] = pos
                    anchor_positions[constellation].pop(anchor)
            
            return anchor_positions

        def get_tag_moment_arms() -> dict[str, dict[int, np.ndarray]]:
            """
            Get tag moment arms in the robot's own body frame.
            
            Args:
            - exp_name: Experiment name.
            
            Returns:
            dict
            - Tag moment arms in the robot's own body frame.
            """
            
            with open(f"config/uwb/tags.yaml", "r") as file:
                tag_moment_arms = yaml.safe_load(file)
            
            for robot in tag_moment_arms:
                tags = list(tag_moment_arms[robot].keys())
                for tag in tags:
                    pos = np.array([eval(tag_moment_arms[robot][tag])]).reshape(3, 1)
                    tag_moment_arms[robot][int(tag)] = pos
                    tag_moment_arms[robot].pop(tag)
            
            return tag_moment_arms

        def zero_order_hold(query_timestamps, data: pd.DataFrame) -> pd.DataFrame:
            """
            Zero-order hold interpolation for data.
            
            Args:
            - query_timestamps: Query timestamps.
            - data: Data to perform zero-order hold interpolation on.
            
            Returns:
            pd.DataFrame
            - New data with zero-order hold interpolation.
            """
            new_data = pd.DataFrame()
            
            # Ensure that query timestamps and data timestamps are sorted in ascending order
            query_timestamps = np.sort(query_timestamps)
            data.sort_values("timestamp", inplace=True)

            # Find the indices associated with the query timestamps using a zero-order hold
            idx_to_keep = []
            most_recent_idx = 0
            
            new_data["timestamp"] = query_timestamps
            for timestamp in query_timestamps:
                while most_recent_idx < len(data) and data["timestamp"].iloc[most_recent_idx] <= timestamp:
                    most_recent_idx += 1
                idx_to_keep.append(most_recent_idx - 1)
                
            # Add the columns at the indices associated with the query timestamps
            for col in data.columns:
                if col == "timestamp":
                    continue
                new_data[col] = data.iloc[idx_to_keep][col].values

            return new_data

        def get_se3_poses(quat: np.ndarray, pos: np.ndarray) -> list[SE3]:
            """
            Get SE3 poses from position and quaternion data.
            
            Args:
            - quat: Quaternion data.
            - pos: Position data.
            
            Returns:
            - SE3 poses.
            """
            
            poses = []
            for i in range(pos.shape[1]):
                R = SO3.from_quat(quat[:, i], "xyzw")
                poses.append(SE3.from_components(R, pos[:, i]))
            return poses

        def get_se23_poses(quat: np.ndarray, vel: np.ndarray, pos: np.ndarray) -> list[SE23]:
            """
            Get SE23 poses from position, velocity, and quaternion data.
            
            Args:
            - quat: Quaternion data.
            - vel: Velocity data.
            - pos: Position data.
            
            Returns:
            - SE23 poses.
            """
            
            poses = []
            for i in range(pos.shape[1]):
                R = SO3.from_quat(quat[:, i], "xyzw")
                poses.append(SE23.from_components(R, vel[:, i], pos[:, i]))
            return poses


        def get_mocap_splines(mocap: pd.DataFrame) -> list[ISmoothingSpline, ISmoothingSpline]:
            """
            Get spline interpolations for mocap data.
            
            Args:
            - mocap: DataFrame containing mocap data.
            
            Returns:
            - pos_splines: Spline interpolation for position.
            - quat_splines: Spline interpolation for orientation.
            """

            # Get mocap data
            time = mocap['timestamp'].values
            pos = mocap[["pose.position.x", "pose.position.y",
                         "pose.position.z"]].values
            quat = mocap[[
                "pose.orientation.x", "pose.orientation.y", "pose.orientation.z",
                "pose.orientation.w"
            ]].values

            # Remove mocap gaps
            pos_gaps = np.linalg.norm(pos, axis=1) < 1e-6
            quat_gaps = np.linalg.norm(quat, axis=1) < 1e-6
            gaps = pos_gaps | quat_gaps

            time = time[~gaps]
            pos = pos[~gaps]
            quat = quat[~gaps]

            # Remove mocap outliers
            outliers = np.zeros(len(time), dtype=bool)
            last_good_R = Rotation.from_quat(quat[0]).as_matrix()
            for i in range(1, len(quat)):
                R_now = Rotation.from_quat(quat[i]).as_matrix()
                
                if (Rotation.from_matrix(last_good_R.T @ R_now).magnitude() > 1):
                    outliers[i-1] = True
                    outliers[i] = True
                else:
                    last_good_R = R_now
                    
            time = time[~outliers]
            pos = pos[~outliers]
            quat = quat[~outliers]

            # Normalize quaternion
            quat /= np.linalg.norm(quat, axis=1)[:, None]

            # Resolve quaternion discontinuities
            for i in range(1, len(quat)):
                if np.dot(quat[i], quat[i - 1]) < 0:
                    quat[i] *= -1

            # Fit splines
            pos_splines = csaps(time, pos.T, smooth=0.9999).spline
            quat_splines = csaps(time, quat.T, smooth=0.9999).spline

            return pos_splines, quat_splines

        def add_imu_bias(
            imu_data: pd.DataFrame,
            pos_spline: ISmoothingSpline, 
            quat_spline: ISmoothingSpline
        ) -> None:
            """
            Get IMU biases.
            
            Args:
            - imu_data: IMU data with the following columns:
                - timestamp
                - angular_velocity.x
                - angular_velocity.y
                - angular_velocity.z
                - linear_acceleration.x
                - linear_acceleration.y
                - linear_acceleration.z
            - pos_spline: Spline interpolation for position.
            - quat_spline: Spline interpolation for orientation.
            
            Returns:
            tuple
            - gyro_bias: Gyroscope bias at the query timestamps.
            - accel_bias: Accelerometer bias at the query timestamps.
            """
            time = imu_data["timestamp"].values
            gyro = np.array([
                imu_data["angular_velocity.x"],
                imu_data["angular_velocity.y"],
                imu_data["angular_velocity.z"],
            ])
            accel = np.array([
                imu_data["linear_acceleration.x"],
                imu_data["linear_acceleration.y"],
                imu_data["linear_acceleration.z"],
            ])
            
            gt_gyro = get_angular_velocity_splines(time, quat_spline)(time)
            gt_accel = get_accelerometer_splines(time, pos_spline, quat_spline)(time)
            
            gyro_bias = np.array([
                csaps(time, gt_gyro[0, :] - gyro[0, :], time, smooth=1e-4), 
                csaps(time, gt_gyro[1, :] - gyro[1, :], time, smooth=1e-4),
                csaps(time, gt_gyro[2, :] - gyro[2, :], time, smooth=1e-4)
            ])
            
            accel_bias = np.array([
                csaps(time, gt_accel[0, :] - accel[0, :], time, smooth=1e-3), 
                csaps(time, gt_accel[1, :] - accel[1, :], time, smooth=1e-3),
                csaps(time, gt_accel[2, :] - accel[2, :], time, smooth=1e-3)
            ])
            
            imu_data["gyro_bias.x"] = gyro_bias[0, :]
            imu_data["gyro_bias.y"] = gyro_bias[1, :]
            imu_data["gyro_bias.z"] = gyro_bias[2, :]
            
            imu_data["accel_bias.x"] = accel_bias[0, :]
            imu_data["accel_bias.y"] = accel_bias[1, :]
            imu_data["accel_bias.z"] = accel_bias[2, :]    

        def get_angular_velocity_splines(time: np.ndarray, quat_splines: ISmoothingSpline) -> ISmoothingSpline:
            """
            Get spline interpolations for angular velocity in the robot's own body frame.
            
            Args:
            - time: Timestamps.
            - quat_splines: Spline interpolations for orientation.
            
            Returns:
            - gyro_splines: Spline interpolations for angular velocity.
            """
            q = quat_splines(time)
            q: np.ndarray = q / np.linalg.norm(q, axis=0)
            N = q.shape[1]
            q_dot = np.atleast_2d(quat_splines.derivative(nu=1)(time)).T
            eta = q[3]
            eps = q[:3]

            S = np.zeros((N, 3, 4))
            for i in range(N):
                e = eps[:, i].reshape((-1, 1))
                S[i, :, :] = np.hstack((2 * (eta[i] * np.eye(3) - SO3.wedge(e)), -2 * e))
                        
            omega = (S @ np.expand_dims(q_dot, 2)).squeeze()
            return csaps(time, omega.T, smooth=0.9).spline

        def get_accelerometer_splines(time: np.ndarray, pos_splines: ISmoothingSpline, quat_splines: ISmoothingSpline) -> ISmoothingSpline:
            """
            Get spline interpolations for accelerometer.
            
            Args:
            - time: Timestamps.
            - pos_splines: Spline interpolations for position.
            - quat_splines: Spline interpolations for orientation.
            
            Returns:
            - accel_splines: Spline interpolations for accelerometer.
            """
            gravity = np.array([0, 0, -9.80665])
            
            q = quat_splines(time)
            acceleration = np.atleast_2d(pos_splines.derivative(nu=2)(time)).T
            
            accelerometer = np.zeros((len(time), 3))
            for i in range(len(time)):
                R = Rotation.from_quat(q[:, i]).as_matrix()
                accelerometer[i] = R.T @ (acceleration[i] - gravity)
                
            return csaps(time, accelerometer.T, smooth=0.99).spline
                
        def get_timeshift(exp_name):
            """
            Get timeshift.
            
            Args:
            - exp_name: Experiment name.
            
            Returns:
            - timeshift: Timeshift in seconds.
            """

            with open(f"data/{exp_name}/timeshift.yaml", "r") as file:
                timeshift = yaml.safe_load(file)
            timeshift_s = timeshift["timeshift_s"]
            timeshift_ns = timeshift["timeshift_ns"]

            return timeshift_s + timeshift_ns / 1e9

        def get_imu_noise_params(robot_name, sensor_name) -> dict:
            """
            Get IMU noise parameters that were generated using allan_variance_ros, available at
            https://github.com/ori-drs/allan_variance_ros. The noise parameters are stored in 
            the config/imu directory.
            
            Args:
            - robot_name: Robot name, e.g., "ifo001".
            - sensor_name: Sensor name, options are "px4" and "cam".
            
            Returns:
            dict
            - gyro: Gyroscope noise parameters.
            - accel: Accelerometer noise parameters.
            - gyro_bias: Gyroscope bias noise parameters.
            - accel_bias: Accelerometer bias noise parameters.
            """
            
            with open(f"config/imu/{robot_name}/{sensor_name}_output.log", "r") as file:
                imu_params = yaml.safe_load(file)
            
            gyro = np.array([
                eval(imu_params["X Angle Random Walk"].split(" ")[0]),
                eval(imu_params["Y Angle Random Walk"].split(" ")[0]),
                eval(imu_params["Z Angle Random Walk"].split(" ")[0])
            ]) * np.pi / 180
            accel = np.array([
                eval(imu_params["X Velocity Random Walk"].split(" ")[0]),
                eval(imu_params["Y Velocity Random Walk"].split(" ")[0]),
                eval(imu_params["Z Velocity Random Walk"].split(" ")[0])
            ])
            
            gyro_bias = np.array([
                eval(imu_params["X Rate Random Walk"].split(" ")[0]),
                eval(imu_params["Y Rate Random Walk"].split(" ")[0]),
                eval(imu_params["Z Rate Random Walk"].split(" ")[0])
            ]) * np.pi / 180
            accel_bias = np.array([
                eval(imu_params["X Accel Random Walk"].split(" ")[0]),
                eval(imu_params["Y Accel Random Walk"].split(" ")[0]),
                eval(imu_params["Z Accel Random Walk"].split(" ")[0])
            ])
            
            return {"gyro": gyro, "accel": accel, "gyro_bias": gyro_bias, "accel_bias": accel_bias}
            
        def get_height_bias(robot_name) -> float:
            """
            The height measurements have a bias, since the height is measured from the ground, and the ground
            is not at the same level as the origin of the motion capture system. This bias between the ground
            and the origin of the motion capture system is returned in this function in meters.
            The bias is subtracted from the height measurements to get an unbiased height measurement.
            
            Args:
            - robot_name: Robot name, e.g., "ifo001".
            
            Returns:
            float
            - Height bias in meters.
            """
            
            with open(f"config/height/bias.yaml", "r") as file:
                height_data = yaml.safe_load(file)
            
            return eval(height_data[robot_name])

        def load_vins(exp_name, robot_id, loop = True, postprocessed: bool = False) -> pd.DataFrame:
            """
            Load VINS data.
            
            Args:
            - exp_name: Experiment name.
            - robot_id: Robot ID.
            - loop: Whether to load VINS data with loop closure or not.
            - postprocessed: Whether to load postprocessed (aligned and shifted) VINS data or not.
            
            Returns:
            - vins: VINS data.
            """
            
            if postprocessed:
                suffix = "_aligned_and_shifted"
            else:
                suffix = ""

            if loop:
                file = f"data/vins/{exp_name}/{robot_id}_vio_loop{suffix}.csv"
            else:
                file = f"data/vins/{exp_name}/{robot_id}_vio{suffix}.csv"

            data = pd.read_csv(
                file,
                names=[
                    "timestamp",
                    "pose.position.x",
                    "pose.position.y",
                    "pose.position.z",
                    "pose.orientation.x",
                    "pose.orientation.y",
                    "pose.orientation.z",
                    "pose.orientation.w",
                    "twist.linear.x",
                    "twist.linear.y",
                    "twist.linear.z",
                ],
                index_col=False,
                header = (0 if postprocessed else None)
            )

            timeshift = get_timeshift(exp_name)
            if not postprocessed:
                data["timestamp"] = data["timestamp"] / 1e9 - timeshift

            return data


        def save_vins(data: pd.DataFrame,
                      exp_name: str,
                      robot_id: str,
                      loop: bool = True,
                      postprocessed: bool = False):
            """
            Save VINS data.
            
            Args:
            - data: VINS data.
            - exp_name: Experiment name.
            - robot_id: Robot ID.
            - loop: Whether loop closure was enabled or not, only affects csv file name.
            - postprocessed: Whether the data is postprocessed or not, only affects csv file name.
            """
            
            if postprocessed:
                suffix = "_aligned_and_shifted"
            else:
                suffix = ""
            
            if loop:
                data.to_csv(f"data/vins/{exp_name}/{robot_id}_vio_loop{suffix}.csv",
                            index=False)
            else:
                data.to_csv(f"data/vins/{exp_name}/{robot_id}_vio{suffix}.csv",
                            index=False)


        def align_frames(df1, df2):
            """
            Align inertial reference frames for two dataframes consisting of body-frame data. 
            The data in the first dataframe is resolved to the inertial reference frame of 
            the second dataframe. The data in the second dataframe is not modified. 
            
            Both dataframes must have measurements at the same timestamps and have the 
            following columns:
            - timestamp
            - pose.position.x
            - pose.position.y
            - pose.position.z
            - pose.orientation.x
            - pose.orientation.y
            - pose.orientation.z
            - pose.orientation.w
            
            Args:
            - df1: First dataframe.
            - df2: Second dataframe.
            
            Returns: dict
            - data: First dataframe with aligned data.
            - C: Rotation matrix from mocap frame to VINS frame.
            - r: Translation vector from mocap frame to VINS frame, resolved in the mocap frame.
            """
            pos1 = df1[[
                "pose.position.x",
                "pose.position.y",
                "pose.position.z",
            ]].values

            pos2 = df2[[
                "pose.position.x",
                "pose.position.y",
                "pose.position.z",
            ]].values

            y = pos1

            C_hat = np.eye(3)
            r_hat = np.zeros(3)

            # Levenberg-Marquardt optimization
            def error(y, C, r):
                return (y - (C @ (pos2 - r).T).T).flatten()

            def jacobian(C, r):
                J = np.empty((0, 6))
                for pos in pos2:
                    J_iter = np.zeros((3, 6))
                    J_iter[:, :3] = -C @ so3_wedge_matrix(pos - r)
                    J_iter[:, 3:] = -C
                    J = np.vstack((J, J_iter))
                return J

            del_x = np.ones(6)
            iter = 0
            e = error(y, C_hat, r_hat)
            while np.linalg.norm(del_x) > 1e-12 and iter < 100:
                J = jacobian(C_hat, r_hat)
                K = np.linalg.inv(J.T @ J + 1e-6 * np.eye(6)) @ J.T
                del_x = K @ e
                r_hat = r_hat + del_x[3:]
                C_hat = C_hat @ sp.linalg.expm(so3_wedge_matrix(del_x[:3]))
                iter += 1

                e = error(y, C_hat, r_hat)
                print("Iteration: ", iter)
                print("Error: ", e)
                print("Error norm: ", np.linalg.norm(e))
                print("Delta x: ", del_x)
                print("Delta x norm: ", np.linalg.norm(del_x))
                print("C_hat: ", C_hat)
                print("r_hat: ", r_hat)
                print("-------------------")

            # Apply transformation to df1
            df1 = apply_transformation(df1, C_hat, r_hat)

            return {"data": df1, "C": C_hat, "r": r_hat}


        def apply_transformation(df, C, r):
            """
            Apply a transformation to a dataframe consisting of body-frame data.
            
            Args:
            - df: Dataframe.
            - C: Rotation matrix.
            - r: Translation vector.
            
            Returns:
            - df: Dataframe with transformed data.
            """

            pose = df[[
                "pose.position.x", "pose.position.y", "pose.position.z",
                "pose.orientation.x", "pose.orientation.y", "pose.orientation.z",
                "pose.orientation.w"
            ]].values

            df_r = pose[:, :3]
            df_quat = pose[:, 3:]
            df_C = np.array([
                Rotation.from_quat(df_quat[i]).as_matrix() for i in range(len(df_quat))
            ])

            pose = np.array([C.T @ df_r[i] + r for i in range(len(df_r))])
            df[[
                "pose.position.x",
                "pose.position.y",
                "pose.position.z",
            ]] = np.array([pose[i] for i in range(len(pose))])
            df[[
                "pose.orientation.x", "pose.orientation.y", "pose.orientation.z",
                "pose.orientation.w"
            ]] = np.array([
                Rotation.from_matrix(df_C[i].T @ C).as_quat() for i in range(len(df_C))
            ])

            if "twist.linear.x" in df.columns:
                df_vel = df[["twist.linear.x", "twist.linear.y",
                             "twist.linear.z"]].values
                vel = np.array([C.T @ df_vel[i] for i in range(len(df_vel))])
                df[["twist.linear.x", "twist.linear.y",
                    "twist.linear.z"]] = np.array([vel[i] for i in range(len(vel))])

            return df


        def so3_wedge_matrix(omega):
            """
            Create a 3x3 SO(3) wedge matrix from a 3x1 vector.
            
            Args:
            - omega: 3x1 vector.
            
            Returns:
            - omega_hat: 3x3 SO(3) cross matrix.
            """

            omega_hat = np.zeros((3, 3))
            omega_hat[0, 1] = -omega[2]
            omega_hat[0, 2] = omega[1]
            omega_hat[1, 0] = omega[2]
            omega_hat[1, 2] = -omega[0]
            omega_hat[2, 0] = -omega[1]
            omega_hat[2, 1] = omega[0]

            return omega_hat


        def compute_position_rmse(df1, df2):
            """
            Compute the root mean squared error (RMSE) between two dataframes consisting of 
            position data.
            
            Args:
            - df1: First dataframe.
            - df2: Second dataframe.
            
            Returns:
            - rmse: RMSE.
            """

            pos1 = df1[["pose.position.x", "pose.position.y",
                        "pose.position.z"]].values

            pos2 = df2[["pose.position.x", "pose.position.y",
                        "pose.position.z"]].values

            return np.sqrt(np.mean(np.linalg.norm(pos1 - pos2, axis=1)**2))
        ```
    *   **__init__.py**:
        ```python
        from .data import DataLoader
        from .utils import *
        ```
*   **paper/**
    *   **calibrate_uwb.py**:
        ```python
        # %%
        from pyuwbcalib.machine import RosMachine
        from pyuwbcalib.postprocess import PostProcess
        from pyuwbcalib.utils import save, set_plotting_env, merge_calib_results
        from pyuwbcalib.uwbcalibrate import UwbCalibrate
        from configparser import ConfigParser, ExtendedInterpolation
        import numpy as np
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib

        # Set the plotting environment
        set_plotting_env()

        # The configuration files
        config_files = [
            'data/bias_calibration_anchors0/config.config',
            'data/bias_calibration_anchors1/config.config',
            'data/bias_calibration_tags0/config.config',
            'data/bias_calibration_tags1/config.config',
        ]

        bias_raw = np.empty(0)
        bias_antenna_delay = np.empty(0)
        bias_fully_calib = np.empty(0)
        calib_results_list = []

        for config in config_files:
            # Parse through the configuration file
            parser = ConfigParser(interpolation=ExtendedInterpolation())
            parser.read(config)

            # Create a RosMachine object for every machine
            machines = {}
            for i,machine in enumerate(parser['MACHINES']):
                machine_id = parser['MACHINES'][machine]
                machines[machine_id] = RosMachine(parser, i)

            # Process and merge the data from all the machines
            data = PostProcess(machines)

            # Instantiate a UwbCalibrate object, and remove static extremes
            calib = UwbCalibrate(data, rm_static=True)

            # Compute the raw bias measurements
            bias_raw = np.append(bias_raw, np.array(calib.df['bias']))

            # Correct antenna delays
            calib.calibrate_antennas(inplace=True, loss='huber')

            # Compute the antenna-delay-corrected measurements
            bias_antenna_delay = np.append(bias_antenna_delay, np.array(calib.df['bias']))

            # Correct power-correlated bias
            calib.fit_power_model(
                inplace = True,
            )

            # Compute the fully-calibrated measurements
            bias_fully_calib = np.append(bias_fully_calib, np.array(calib.df['bias']))

            # Save the calibration results
            calib_results = {
                'delays': calib.delays,
                'bias_spl': calib.bias_spl,
                'std_spl': calib.std_spl,
            }
            save(
                calib_results, 
                config.split('/config')[0] + '/calib_results.pickle'
            )
            
            calib_results_list.append(calib_results)
            
        calib_results = merge_calib_results(calib_results_list)

        save(
            calib_results, 
            "config/uwb_calib.pickle"
        )

        plt.rc('legend', fontsize=40)
        print(calib_results['delays'])

        fig, axs = plt.subplots(2, 1, sharex=True)
        x = np.linspace(0, 1.5)
        axs[0].plot(x, calib.bias_spl(x)*100, label='Bias')
        axs[1].plot(x, calib.std_spl(x)*100, label='Standard deviation')
        axs[0].set_ylabel('Bias [cm]')
        axs[0].set_yticks([-10, -5, 0, 5, 10])
        axs[1].set_ylabel('Bias Std. [cm]')
        axs[1].set_xlabel("Lifted signal strength")
        axs[1].set_yticks([0, 10, 20])
        axs[1].set_xticks(np.arange(0, 1.6, 0.2))

        bins = 200
        fig2 = plt.figure()
        plt.hist(bias_raw, density=True, bins=bins, alpha=0.5, label='Raw')
        plt.hist(bias_antenna_delay, density=True, bins=bins, alpha=0.5, label='Antenna-delay calibrated')
        plt.hist(bias_fully_calib, density=True, bins=bins, alpha=0.5, label='Fully calibrated')
        plt.xticks(np.arange(-0.4, 1, 0.2))
        plt.xlabel('Bias [m]')
        plt.xlim([-0.5, 1])
        plt.legend()

        fig.savefig('figs/calib_results.pdf')
        fig2.savefig('figs/bias_histogram.pdf')

        plt.show()
        ```
    *   **ekf_all.py**:
        ```python
        # %%
        import pandas as pd
        import subprocess

        from joblib import Parallel, delayed

        def call_vins_ekf(exp_name, num_robots):
            print("Running VINS EKF for experiment", exp_name)
            
            if num_robots == 1:
                subprocess.run(["python", "examples/ekf_vins_one_robot.py", exp_name])
            elif num_robots == 3:
                subprocess.run(["python", "examples/ekf_vins_three_robots.py", exp_name])
                
        def call_imu_ekf(exp_name, num_robots):
            print("Running IMU EKF for experiment", exp_name)
            
            if num_robots == 1:
                subprocess.run(["python", "examples/ekf_imu_one_robot.py", exp_name])
            elif num_robots == 3:
                subprocess.run(["python", "examples/ekf_imu_three_robots.py", exp_name])
            

        if __name__ == "__main__":
            data = pd.read_csv("config/experiments.csv")

            tasks = []
            for i in range(len(data)):
                num_robots = data["num_robots"].iloc[i]
                exp_name = str(data["experiment"].iloc[i])
                tasks.append(delayed(call_vins_ekf)(exp_name, num_robots))
                tasks.append(delayed(call_imu_ekf)(exp_name, num_robots))

            # Run tasks in parallel
            Parallel(n_jobs=-1)(tasks)
        ```
    *   **evaluate_vins.py**:
        ```python
        # %%
        from miluv.data import DataLoader
        from miluv.utils import load_vins, align_frames, compute_position_rmse, save_vins, apply_transformation
        import pandas as pd
        import matplotlib.pyplot as plt
        import sys
        import yaml
        from scipy.spatial.transform import Rotation

        def evaluate_vins(exp_name, robot_id, visualize):
            # Read sensor and mocap data
            mv = DataLoader(exp_name, barometer=False, cir=False)

            # Read vins data
            vins = load_vins(exp_name, robot_id)

            # Drop the last 10 seconds of vins data
            vins = vins[vins["timestamp"] < vins["timestamp"].iloc[-1] - 10]

            # Get mocap data at vins timestamps
            pos = mv.data[robot_id]["mocap_pos"](vins["timestamp"])
            quat = mv.data[robot_id]["mocap_quat"](vins["timestamp"])

            # Align frame
            df_mocap = pd.DataFrame({
                "timestamp": vins["timestamp"],
                "pose.position.x": pos[0],
                "pose.position.y": pos[1],
                "pose.position.z": pos[2],
                "pose.orientation.x": quat[0],
                "pose.orientation.y": quat[1],
                "pose.orientation.z": quat[2],
                "pose.orientation.w": quat[3],
            })
            results = align_frames(vins, df_mocap)
            vins = results["data"]
            frame_alignment = {
                "phi_vm": Rotation.from_matrix(results["C"]).as_rotvec().tolist(),
                "r_vm_m": results["r"].tolist(),
            }
            
            # Save frame_alignment to a yaml file
            with open(f"data/vins/{exp_name}/{robot_id}_alignment_pose.yaml", "w") as file:
                yaml.dump(frame_alignment, file)
            save_vins(vins, exp_name, robot_id, postprocessed=True)
            
            rmse_loop = compute_position_rmse(vins, df_mocap)
            print(f"Position RMSE w Loop Closure for \Experiment {exp_name} \
                                                and Robot {robot_id}: {rmse_loop} m")
            
            # Apply transformation to vins without loop closure
            vins_no_loop = load_vins(exp_name, robot_id, loop=False)
            vins_no_loop = vins_no_loop[vins_no_loop["timestamp"] < vins_no_loop["timestamp"].iloc[-1] - 10]
            pos_no_loop = mv.data[robot_id]["mocap_pos"](vins_no_loop["timestamp"])
            quat_no_loop = mv.data[robot_id]["mocap_quat"](vins_no_loop["timestamp"])
            df_mocap_no_loop = pd.DataFrame({
                "timestamp": vins_no_loop["timestamp"],
                "pose.position.x": pos_no_loop[0],
                "pose.position.y": pos_no_loop[1],
                "pose.position.z": pos_no_loop[2],
                "pose.orientation.x": quat_no_loop[0],
                "pose.orientation.y": quat_no_loop[1],
                "pose.orientation.z": quat_no_loop[2],
                "pose.orientation.w": quat_no_loop[3],
            })
            vins_no_loop = apply_transformation(vins_no_loop, results["C"], results["r"])
            save_vins(vins_no_loop, exp_name, robot_id, loop=False, postprocessed=True)
            rmse_no_loop = compute_position_rmse(vins_no_loop, df_mocap_no_loop)
            print(f"Position RMSE w/o Loop Closure for Experiment {exp_name} \
                                                and Robot {robot_id}: {rmse_no_loop} m")

            if visualize:
                # Compare vins and mocap data
                fig = plt.figure()
                fig.suptitle("VINS vs. Mocap Position w Loop Closure")
                ax = plt.axes(projection ='3d')
                ax.plot3D(vins["pose.position.x"], vins["pose.position.y"], vins["pose.position.z"], label="vins")
                ax.plot3D(pos[0], pos[1], pos[2], label="mocap")
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.set_zlabel("z [m]")
                ax.legend()
                ax.grid()   

                fig, axs = plt.subplots(3, 1)
                fig.suptitle("VINS vs. Mocap Position w Loop Closure")
                axs[0].plot(vins["timestamp"], vins["pose.position.x"], label="vins")
                axs[0].plot(vins["timestamp"], pos[0], label="mocap")
                axs[0].set_ylabel("x [m]")
                axs[1].plot(vins["timestamp"], vins["pose.position.y"], label="vins")
                axs[1].plot(vins["timestamp"], pos[1], label="mocap")
                axs[1].set_ylabel("y [m]")
                axs[2].plot(vins["timestamp"], vins["pose.position.z"], label="vins")
                axs[2].plot(vins["timestamp"], pos[2], label="mocap")
                axs[2].set_ylabel("z [m]")
                plt.legend()
                axs[0].grid()
                axs[1].grid()
                axs[2].grid()

                fig, axs = plt.subplots(3, 1)
                fig.suptitle("VINS vs. Mocap Position Error w Loop Closure")
                axs[0].plot(vins["timestamp"], vins["pose.position.x"] - pos[0], label="x")
                axs[0].set_ylabel("x [m]")
                axs[1].plot(vins["timestamp"], vins["pose.position.y"] - pos[1], label="y")
                axs[1].set_ylabel("y [m]")
                axs[2].plot(vins["timestamp"], vins["pose.position.z"] - pos[2], label="z")
                axs[2].set_ylabel("z [m]")
                axs[0].grid()
                axs[1].grid()
                axs[2].grid()
                
                # Do the same for vins without loop closure
                pos_no_loop = mv.data[robot_id]["mocap_pos"](vins_no_loop["timestamp"])
                
                fig = plt.figure()
                fig.suptitle("VINS vs. Mocap Position w/o Loop Closure")
                ax = plt.axes(projection ='3d')
                ax.plot3D(vins_no_loop["pose.position.x"], vins_no_loop["pose.position.y"], vins_no_loop["pose.position.z"], label="vins")
                ax.plot3D(pos_no_loop[0], pos_no_loop[1], pos_no_loop[2], label="mocap")
                ax.set_xlabel("x [m]")
                ax.set_ylabel("y [m]")
                ax.set_zlabel("z [m]")
                ax.legend()
                ax.grid()
                
                fig, axs = plt.subplots(3, 1)
                fig.suptitle("VINS vs. Mocap Position w/o Loop Closure")
                axs[0].plot(vins_no_loop["timestamp"], vins_no_loop["pose.position.x"], label="vins")
                axs[0].plot(vins_no_loop["timestamp"], pos_no_loop[0], label="mocap")
                axs[0].set_ylabel("x [m]")
                axs[1].plot(vins_no_loop["timestamp"], vins_no_loop["pose.position.y"], label="vins")
                axs[1].plot(vins_no_loop["timestamp"], pos_no_loop[1], label="mocap")
                axs[1].set_ylabel("y [m]")
                axs[2].plot(vins_no_loop["timestamp"], vins_no_loop["pose.position.z"], label="vins")
                axs[2].plot(vins_no_loop["timestamp"], pos_no_loop[2], label="mocap")
                axs[2].set_ylabel("z [m]")
                plt.legend()
                axs[0].grid()
                axs[1].grid()
                axs[2].grid()

                fig, axs = plt.subplots(3, 1)
                fig.suptitle("VINS vs. Mocap Position Error w/o Loop Closure")
                axs[0].plot(vins_no_loop["timestamp"], vins_no_loop["pose.position.x"] - pos_no_loop[0], label="x")
                axs[0].set_ylabel("x [m]")
                axs[1].plot(vins_no_loop["timestamp"], vins_no_loop["pose.position.y"] - pos_no_loop[1], label="y")
                axs[1].set_ylabel("y [m]")
                axs[2].plot(vins_no_loop["timestamp"], vins_no_loop["pose.position.z"] - pos_no_loop[2], label="z")
                axs[2].set_ylabel("z [m]")
                axs[0].grid()
                axs[1].grid()
                axs[2].grid()

                plt.show(block=True)
                
            return {"rmse_loop": rmse_loop, "rmse_no_loop": rmse_no_loop}

        if __name__ == "__main__":
            if len(sys.argv) < 3:
                print("Not enough arguments. Usage: python evaluate_vins.py exp_name robot_id")
                sys.exit(1)
            exp_name = sys.argv[1]
            robot_id = sys.argv[2]
            
            if len(sys.argv) == 4:
                visualize = sys.argv[3]
            else:
                visualize = False
            
            evaluate_vins(exp_name, robot_id, visualize)
        ```
    *   **extract_image.py**:
        ```python
        # %%
        import matplotlib.pyplot as plt
        from cv_bridge import CvBridge
        import rosbag 
        import numpy.random as random
        import numpy as np

        # from pyuwbcalib.utils import set_plotting_env
        # set_plotting_env()
        np.random.seed(5)
        input_bag = '/home/shalaby/Desktop/datasets/miluv_dataset/main/calib/ifo001/ifo001_calib1_2024-02-12-09-15-18.bag'
        # b = bagreader(input_bag)

        # df = pd.read_csv(b.message_by_topic("/ifo001/camera/color/image_raw/compressed"))

        cv_image = []
        t_image = []

        bridge = CvBridge()
        idx_list = random.randint(0, 200000, 9)
        idx_list.sort()
        i = 0 
        for topic, msg, t in rosbag.Bag(input_bag).read_messages():
            if i == 0:
                t0 = t.to_sec()
            if i < idx_list[0]:
                i += 1
                continue
            if topic != "/ifo001/camera/color/image_raw/compressed":
                continue
            cv_image = cv_image + [bridge.compressed_imgmsg_to_cv2(msg)]
            t_image = t_image + [t.to_sec()]
            idx_list = idx_list[1:]
            
            if len(idx_list) == 0:
                break

        # print(t_image)
        fig, axs = plt.subplots(3, 3)
        for i in range(3):
            for j in range(3):
                axs[i, j].imshow(cv_image[i*3+j])
                axs[i, j].axis('off')
                axs[i,j].set_title("t = " + str(np.round(t_image[i*3+j] - t0)) + " s")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')

        plt.savefig("figs/kalibr_calib.pdf")

        plt.show()
        ```
    *   **vins_all.py**:
        ```python
        # %%
        import subprocess
        import pandas as pd
        from paper.evaluate_vins import evaluate_vins

        def call_vins(exp_name, robot_id):
            print("Running VINS for experiment", exp_name, "robot", robot_id)
            subprocess.run(["./examples/run_vins.sh", exp_name, robot_id])
            
            return evaluate_vins(exp_name, robot_id, False)
            

        if __name__ == "__main__":
            data = pd.read_csv("config/experiments.csv")
            rmse_df = pd.DataFrame(columns=["experiment", "robot", "rmse_loop", "rmse_no_loop"])
            for i in range(len(data)):
                num_robots = data["num_robots"].iloc[i]
                for j in range(num_robots):
                    exp_name = str(data["experiment"].iloc[i])
                    rmse = call_vins(exp_name, f"ifo00{j+1}")
                    new_df = pd.DataFrame({
                        "experiment": [exp_name],
                        "robot": [f"ifo00{j+1}"],
                        "rmse_loop": [rmse["rmse_loop"]],
                        "rmse_no_loop": [rmse["rmse_no_loop"]]
                    })
                    rmse_df = pd.concat([rmse_df, new_df], ignore_index=True)
                    
                    rmse_df.to_csv("data/vins/vins_rmse.csv", index=False)            
        ```
    *   **visualize_cir.py**:
        ```python
        # %%
        import numpy as np
        from bagpy import bagreader
        import matplotlib.pyplot as plt
        import pandas as pd

        from pyuwbcalib.utils import set_plotting_env

        set_plotting_env()

        b = bagreader('/home/shalaby/Desktop/datasets/miluv_dataset/main/12/ifo001_exp12c_2024-02-01-13-21-48.bag')

        df = pd.read_csv(b.message_by_topic("/ifo001/uwb/cir"))
        cir_cols = df.columns[df.columns.str.contains('cir')]
        df["cir"] = df[cir_cols].values.tolist()
        df = df.drop(columns=cir_cols)

        k = 105
        plt.plot(df.iloc[k].cir, label="CIR")
        plt.plot(np.ones(2)*df.iloc[k].idx, [0,8000], linewidth=3, label="Peak")
        plt.ylabel("Amplitude ")
        plt.xlabel("Sample index")
        plt.xlim([675, 900])
        plt.ylim([0, 7000])
        plt.legend()

        plt.savefig("figs/cir_example.pdf")
        ```
*   **preprocess/**
    *   **clean.sh**:
        ```bash
        #!/bin/bash

        CSV_FILE="config/experiments.csv"

        tail -n +2 "$CSV_FILE" | while IFS=, read -r exp _; do
            rm -rf "data/$exp/ifo001"
            rm -rf "data/$exp/ifo002"
            rm -rf "data/$exp/ifo003"
            rm -rf "data/$exp/timeshift.yaml"
        done
        ```
    *   **cleanup_csv.py**:
        ```python
        import sys
        from os import listdir, remove, walk, rename
        from os.path import join, isfile
        import pandas as pd
        import yaml

        # headers to keep for every file
        imu = [
            "timestamp",
            "angular_velocity.x",
            "angular_velocity.y",
            "angular_velocity.z",
            "linear_acceleration.x",
            "linear_acceleration.y",
            "linear_acceleration.z",
        ]

        mag = [
            "timestamp",
            "magnetic_field.x",
            "magnetic_field.y",
            "magnetic_field.z",
        ]
        height = [
            "timestamp",
            "range",
        ]

        mocap = [
            "timestamp",
            "pose.position.x",
            "pose.position.y",
            "pose.position.z",
            "pose.orientation.x",
            "pose.orientation.y",
            "pose.orientation.z",
            "pose.orientation.w",
        ]
        barometer = [
            "timestamp",
            "fluid_pressure",
        ]

        cir = [
            "timestamp",
            "my_id",
            "from_id",
            "to_id",
            "idx",
            "cir",
        ]


        def cleanup_csvs(dir):
            # Find all csv files
            files = [f for f in listdir(dir) if f.endswith('.csv')]

            for file in files:
                if "imu" in file and "camera" in file:
                    process_csv(dir, file, imu, "imu_cam")
                elif "mag" in file and file != "mag.csv":
                    process_csv(dir, file, mag, "mag")
                elif "hrlv" in file:
                    process_csv(dir, file, height, "height")
                elif "vrpn" in file:
                    process_csv(dir, file, mocap, "mocap")
                elif "static_pressure" in file:
                    process_csv(dir, file, barometer, "barometer")
                elif "imu" in file and "mavros" in file and "raw" in file:
                    process_csv(dir, file, imu, "imu_px4")
                elif "uwb" in file and "cir" in file:
                    process_cir(dir, file, cir)


        def process_csv(dir, file, headers, name):
            df = pd.read_csv(join(dir, file))
            df = merge_time(df)
            df = df[headers]
            df.to_csv(join(dir, name + ".csv"), index=False)
            remove(join(dir, file))


        def process_cir(dir, file, headers):
            df = pd.read_csv(join(dir, file))
            df = merge_time(df)
            cir_headers = [f"cir_{int(i)}" for i in range(1016)]
            df["cir"] = df[cir_headers].values.tolist()
            df = df[headers]
            df.to_csv(join(dir, "uwb_cir.csv"), index=False)
            remove(join(dir, file))


        def merge_time(df):
            sec = df["header.stamp.secs"]
            nsec = df["header.stamp.nsecs"]
            df["timestamp"] = sec + nsec / 1e9
            return df


        def find_min_timestamp(all_files):
            """Find the minimum timestamp in all csv files."""
            min_timestamp = float('inf')
            for file in all_files:
                if file.endswith('.csv'):
                    df = pd.read_csv(file)
                    if df["timestamp"].min() < min_timestamp:
                        min_timestamp = df["timestamp"].min()
                elif file.endswith('.jpeg'):
                    img_timestamp = int(file.split(".")[0].split("/")[-1]) / 1e9
                    if img_timestamp < min_timestamp:
                        min_timestamp = img_timestamp
               
            return min_timestamp


        def shift_timestamps(path):
            """Shift all timestamps by the minimum timestamp."""
            all_csvs = []
            all_jpegs = []
            for subdir, dirs, files in walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        all_csvs.append(join(subdir, file))
                    elif file.endswith('.jpeg'):
                        all_jpegs.append(join(subdir, file))

            min_timestamp = find_min_timestamp(all_csvs)
            for file in all_csvs:
                df = pd.read_csv(file)
                df["timestamp"] = df["timestamp"] - min_timestamp
                if "timestamp_n" in df.columns:
                    df["timestamp_n"] = df["timestamp_n"] - min_timestamp
                df.to_csv(file, index=False)
            for file in all_jpegs:
                img_timestamp = int(file.split(".")[0].split("/")[-1]) / 1e9 - min_timestamp
                if img_timestamp < 0:
                    remove(file)
                else:
                    rename(
                        file, 
                        "/".join(file.split("/")[:-1]) + "/" + str(img_timestamp) + ".jpeg"
                    )

            # Save timeshift to yaml file        
            if not isfile(path + "/timeshift.yaml"):
                seconds = int(min_timestamp)
                nanoseconds = int((min_timestamp - seconds) * 1e9)
                with open(path + "/timeshift.yaml", 'w') as file:
                    yaml.dump(
                        {'timeshift_s': seconds, 'timeshift_ns': nanoseconds}, 
                        file,
                        default_flow_style=False
                    )


        if __name__ == '__main__':

            if len(sys.argv) != 2:
                print(
                    "Not enough arguments. Usage: python cleanup_csv.py path_to_csvs")
                sys.exit(1)

            path = sys.argv[1]
            if path.endswith('/'):
                path = path[:-1]
                
            files = [f for f in listdir(path) if f.endswith('.bag')]

            for file in files:
                cleanup_csvs(join(path, file.split(".")[0]))

            shift_timestamps(path)
        ```
    *   **preprocess.sh**:
        ```bash
        #!/bin/bash

        CSV_FILE="config/experiments.csv"

        task(){
            python preprocess/read_bags.py data/$exp True
            python preprocess/process_uwb.py data/$exp
            python preprocess/cleanup_csv.py data/$exp
        }

        N=16
        (
        tail -n +2 "$CSV_FILE" | while IFS=, read -r exp _; do
            rm -rf "data/$exp/ifo001"
            rm -rf "data/$exp/ifo002"
            rm -rf "data/$exp/ifo003"
            rm -rf "data/$exp/timeshift.yaml"
            ((i=i%N)); ((i++==0)) && wait
            task $exp &
        done
        )
        ```
    *   **process_uwb.py**:
        ```python
        # %%
        from pyuwbcalib.machine import RosMachine
        from pyuwbcalib.postprocess import PostProcess
        from pyuwbcalib.utils import load, read_anchor_positions
        from pyuwbcalib.uwbcalibrate import ApplyCalibration
        import sys
        from os.path import join
        import os
        import pandas as pd
        import yaml


        def get_experiment_info(path):
            exp_name = path.split('/')[-1]
            df = pd.read_csv(join("config", "experiments.csv"))
            df["experiment"] = df["experiment"].astype(str)
            row: pd.DataFrame = df[df["experiment"] == exp_name]
            return row.to_dict(orient="records")[0]


        def get_anchors(anchor_constellation):
            with open('config/uwb/anchors.yaml', 'r') as file:
                return yaml.safe_load(file)[anchor_constellation]


        def generate_config(exp_info):
            params = {
                "max_ts_value": "2**32",
                "ts_to_ns": "1e9 * (1.0 / 499.2e6 / 128.0)",
                "ds_twr": "True",
                "passive_listening": "True",
                "fpp_exists": "True",
                "rxp_exists": "False",
                "std_exists": "False",
            }

            pose_path = {
                "directory": f"data/{exp_info['experiment']}/",
            }
            for i in range(exp_info["num_robots"]):
                pose_path.update({f"{i}": f"ifo00{i+1}.bag"})

            uwb_path = pose_path.copy()
            if exp_info["num_anchors"] > 0:
                anchors = get_anchors(str(exp_info["anchor_constellation"]))
            else:
                anchors = None
            machines = {}
            for i in range(exp_info["num_robots"]):
                machines.update({f"{i}": f"ifo00{i+1}"})

            tags = {}
            for i in range(exp_info["num_robots"]):
                if exp_info["num_tags_per_robot"] == 2:
                    tags.update({f"{i}": f"[{(i+1)*10}, {(i+1)*10 + 1}]"})
                elif exp_info["num_tags_per_robot"] == 1:
                    tags.update({f"{i}": f"[{(i+1)*10}]"})

            moment_arms = {
                "10": "[0.13189,-0.17245,-0.05249]",
                "11": "[-0.17542,0.15712,-0.05307]",
                "20": "[0.16544,-0.15085,-0.03456]",
                "21": "[-0.15467,0.16972,-0.01680]",
                "30": "[0.16685,-0.18113,-0.05576]",
                "31": "[-0.13485,0.15468,-0.05164]",
            }

            pose_topic = {}
            for i in range(exp_info["num_robots"]):
                pose_topic.update(
                    {f"{i}": f"/ifo00{i+1}/vrpn_client_node/ifo00{i+1}/pose"})

            uwb_topic = {}
            for i in range(exp_info["num_robots"]):
                uwb_topic.update({f"{i}": f"/ifo00{i+1}/uwb/range"})

            listening_topic = {}
            for i in range(exp_info["num_robots"]):
                listening_topic.update({f"{i}": f"/ifo00{i+1}/uwb/passive"})

            uwb_message = {
                "from_id": "from_id",
                "to_id": "to_id",
                "tx1": "tx1",
                "rx1": "rx1",
                "tx2": "tx2",
                "rx2": "rx2",
                "tx3": "tx3",
                "rx3": "rx3",
                "fpp1": "fpp1",
                "fpp2": "fpp2",
            }

            listening_message = {
                "my_id": "my_id",
                "from_id": "from_id",
                "to_id": "to_id",
                "covariance": "covariance",
                "rx1": "rx1",
                "rx2": "rx2",
                "rx3": "rx3",
                "tx1_n": "tx1_n",
                "rx1_n": "rx1_n",
                "tx2_n": "tx2_n",
                "rx2_n": "rx2_n",
                "tx3_n": "tx3_n",
                "rx3_n": "rx3_n",
                "fpp1": "pr1",
                "fpp2": "pr2",
                "fpp3": "pr3",
                "fpp1_n": "pr1_n",
                "fpp2_n": "pr2_n ",
            }

            return {
                "PARAMS": params,
                "POSE_PATH": pose_path,
                "UWB_PATH": uwb_path,
                "ANCHORS": anchors,
                "MACHINES": machines,
                "TAGS": tags,
                "MOMENT_ARMS": moment_arms,
                "POSE_TOPIC": pose_topic,
                "UWB_TOPIC": uwb_topic,
                "LISTENING_TOPIC": listening_topic,
                "UWB_MESSAGE": uwb_message,
                "LISTENING_MESSAGE": listening_message
            }


        def process_uwb(path):
            # The configuration files
            exp_info = get_experiment_info(path)
            uwb_config = generate_config(exp_info)

            # Read anchor positions
            if exp_info["num_anchors"] > 0:
                anchor_positions = read_anchor_positions(uwb_config)

            # Create a RosMachine object for every machine
            machines = {}
            for i, machine in enumerate(uwb_config['MACHINES']):
                machine_id = uwb_config['MACHINES'][machine]
                machines[machine_id] = RosMachine(uwb_config, i)

            # Process and merge the data from all the machines
            if exp_info["num_anchors"] > 0:
                data = PostProcess(machines, anchor_positions)
            else:
                data = PostProcess(machines)

            # Load the UWB calibration results
            calib_results = load("config/uwb/uwb_calib.pickle", )

            # Apply the calibration
            df = data.df
            df["range_raw"] = df["range"]
            df["bias_raw"] = df["bias"]
            df["tx1_raw"] = df["tx1"]
            df["tx2_raw"] = df["tx2"]
            df["tx3_raw"] = df["tx3"]
            df["rx1_raw"] = df["rx1"]
            df["rx2_raw"] = df["rx2"]
            df["rx3_raw"] = df["rx3"]
            df = ApplyCalibration.antenna_delays(df,
                                                 calib_results["delays"],
                                                 max_value=1e9 *
                                                 (1.0 / 499.2e6 / 128.0) * 2.0**32)
            df = ApplyCalibration.power(df,
                                        calib_results["bias_spl"],
                                        calib_results["std_spl"],
                                        max_value=1e9 * (1.0 / 499.2e6 / 128.0) *
                                        2.0**32)

            df_passive = data.df_passive
            df_passive["rx1_raw"] = df_passive["rx1"]
            df_passive["rx2_raw"] = df_passive["rx2"]
            df_passive["rx3_raw"] = df_passive["rx3"]
            df_passive = ApplyCalibration.antenna_delays_passive(
                df_passive, calib_results["delays"])
            df_passive = ApplyCalibration.power_passive(df_passive,
                                                        calib_results["bias_spl"],
                                                        calib_results["std_spl"])

            # Convert timestamps from seconds to nanoseconds
            df["timestamp"] = df["time"]
            df.drop(columns=["time"], inplace=True)
            df_passive["timestamp"] = df_passive["time"]
            df_passive.drop(columns=["time"], inplace=True)

            # Add back important info to df_passive
            df_iter = df.iloc[df_passive["idx"]]
            to_copy = [
                "tx1", "rx1", "tx2", "rx2", "tx3", "rx3", "range", "bias", "tx1_raw",
                "rx1_raw", "tx2_raw", "rx2_raw", "tx3_raw", "rx3_raw", "range_raw",
                "bias_raw", "gt_range", "timestamp"
            ]
            for col in to_copy:
                df_passive[col + "_n"] = df_iter[col].values

            # Drop unnecessary columns
            df.drop(columns=[
                "header.seq", "header.frame_id", "covariance", "tof1", "tof2", "tof3",
                "sum_t1", "sum_t2"
            ],
                    inplace=True)
            df_passive.drop(
                columns=["header.seq", "header.frame_id", "covariance", "idx"],
                inplace=True)

            # Separate for each robot and save csvs
            for robot in data.tag_ids:
                tags = data.tag_ids[robot]
                robot_init_bool = df["from_id"].isin(tags)
                robot_targ_bool = df["to_id"].isin(tags)
                df_robot = df[robot_init_bool | robot_targ_bool]
                df_robot.to_csv(join(path, f"{robot}/uwb_range.csv"), index=False)

                robot_init_bool = df_passive["from_id"].isin(tags)
                robot_targ_bool = df_passive["to_id"].isin(tags)
                df_robot = df_passive[robot_init_bool | robot_targ_bool]
                df_robot.to_csv(join(path, f"{robot}/uwb_passive.csv"), index=False)


        if __name__ == '__main__':

            if len(sys.argv) != 2:
                print("Not enough arguments. Usage: python cleanup_csv.py path_to_csvs")
                sys.exit(1)
            path = sys.argv[1]
            
            if path.endswith('/'):
                path = path[:-1]

            process_uwb(path)

            # Remove the bagreader-generated UWB csv files
            robots = [f for f in os.listdir(path) if f.endswith('.bag')]
            for robot in robots:
                robot_id = robot.split('.')[0]
                robot_folder = os.path.join(path, robot_id)
                for file in os.listdir(robot_folder):
                    file_path = os.path.join(robot_folder, file)
                    if robot_id in file and "uwb" in file and "cir" not in file:
                        os.remove(file_path)
        ```
    *   **read_apriltags.py**:
        ```python
        # %% 
        import xml.etree.ElementTree as ET
        import pandas as pd
        import yaml

        def read_vsk(
            fname, 
            marker_prefix="apriltags_", 
            parameter_prefix="apriltags_apriltags"
        ):
            out_data = []
            tree = ET.parse(fname)
            root = tree.getroot()
            
            # Get order of marker IDs
            marker_ids = []
            for parent in root.findall("MarkerSet"):  
                for child in parent.findall("Markers"):
                    for grandchild in child.findall("Marker"):
                        name = grandchild.attrib["NAME"]
                        try:
                            id = int(name.split(marker_prefix)[1])
                        except Exception as e:
                            print(f"Error parsing name {name} using marker_prefix {marker_prefix}")
                            print(e)
                            continue
                        marker_ids.append(id)
            
            # Get marker positions
            i = 0
            for child in root.findall("Parameters"):  # Parameters,
                for grandchild in child.findall("Parameter"):  # Parameters,
                    name = grandchild.attrib["NAME"]
                    try:
                        family, rest = name.split(parameter_prefix)
                    except Exception as e:
                        print(f"Error parsing name {name} using parameter_prefix {parameter_prefix}")
                        print(e)
                        continue
                    id_, coord = rest.split("_")
                    id_ = str(marker_ids[i])
                    value = 1e-3 * float(grandchild.attrib["VALUE"])
                    out_data.append(dict(family=family, id=id_, coord=coord, value=value))
                    if coord == "z":
                        i += 1
            df = pd.DataFrame(out_data)
            pt = pd.pivot_table(data=df, index="id", values="value", columns="coord")
            pt["position"] = pt.apply(lambda row: str([row["x"], row["y"], row["z"]]), axis=1)
            pt.drop(columns=["x", "y", "z"], inplace=True)
            return pt.sort_index()

        # Configuration 0
        data_0_part1 = read_vsk(
            "config/setup/apriltags_0_part1_v1.vsk",
            parameter_prefix="apriltags_00_apriltags_00"
        )
        data_0_part2 = read_vsk(
            "config/setup/apriltags_0_part2_v1.vsk", 
            parameter_prefix="apriltags_00_1_apriltags_00_1"
        )
        data_0 = pd.concat([data_0_part1, data_0_part2]).sort_index()

        # Configuration 0b
        data_0b = read_vsk(
            "config/setup/apriltags_0_v2.vsk",
            marker_prefix="april_tags_",
            parameter_prefix="april_tags_0_v2_april_tags_0_v2"
        )

        # Configuration 1
        data_1_part1 = read_vsk(
            "config/setup/apriltags_1_part1.vsk",
            marker_prefix="april_tags_",
            parameter_prefix="april_tags_0_v2_april_tags_0_v2"
        )
        data_1_part2 = read_vsk(
            "config/setup/apriltags_1_part2.vsk",
            parameter_prefix="apriltags_1_apriltags_1"
        )
        data_1 = pd.concat([data_1_part1, data_1_part2]).sort_index()

        # Configuration 2
        data_2_part1 = read_vsk(
            "config/setup/apriltags_2_part1.vsk",
            marker_prefix="april_tags_",
            parameter_prefix="april_tags_0_v2_april_tags_0_v2"
        )
        data_2_part2 = read_vsk(
            "config/setup/apriltags_2_part2.vsk",
            parameter_prefix="apriltags_3_apriltags_3"
        )
        data_2 = pd.concat([data_2_part1, data_2_part2]).sort_index()

        # Save to yaml file
        data = {
            "0": data_0.to_dict()["position"],
            "0b": data_0b.to_dict()["position"],
            "1": data_1.to_dict()["position"],
            "2": data_2.to_dict()["position"]
        }
        with open('config/apriltags/apriltags.yaml', 'w') as file:
            yaml.dump(data, file)
        ```
    *   **read_bags.py**:
        ```python
        import sys
        from os import listdir, mkdir, rename
        from os.path import join, isdir
        import rosbag
        import cv2
        from cv_bridge import CvBridge
        from bagpy import bagreader
        import sys


        def rename_files(files, path):
            new_files = []
            for file in files:
                if "ifo001" in file:
                    rename(join(path, file), join(path, "ifo001.bag"))
                    new_files.append("ifo001.bag")
                elif "ifo002" in file:
                    rename(join(path, file), join(path, "ifo002.bag"))
                    new_files.append("ifo002.bag")
                elif "ifo003" in file:
                    rename(join(path, file), join(path, "ifo003.bag"))
                    new_files.append("ifo003.bag")
            return new_files


        def write_imgs(input_bag, dir_main):
            bridge = CvBridge()
            for topic, msg, _ in rosbag.Bag(input_bag).read_messages():
                if msg._type == 'sensor_msgs/CompressedImage':
                    if "infra1" in topic:
                        dir = dir_main + "infra1/"
                    elif "infra2" in topic:
                        dir = dir_main + "infra2/"
                    elif "bottom" in topic:
                        dir = dir_main + "bottom/"
                    elif "color" in topic:
                        dir = dir_main + "color/"
                    try:
                        cv_image = bridge.compressed_imgmsg_to_cv2(msg)
                        cv2.imwrite(dir + str(msg.header.stamp) + '.jpeg', cv_image)
                    except Exception as e:
                        Warning('Error uncompressing image: {}'.format(e))


        def write_csvs(input_bag):
            b = bagreader(input_bag)
            for topic in b.topics:
                if "camera" in topic and "imu" not in topic:
                    continue
                else:
                    b.message_by_topic(topic)


        if __name__ == '__main__':
            # TODO: Allow user-defined image compression type
            if len(sys.argv) < 2:
                print("Not enough arguments. Usage: python read_bags.py path_to_bags")
                sys.exit(1)
            if len(sys.argv) < 3:
                vision = True
            else:
                vision = eval(sys.argv[2])

            path = sys.argv[1]
            if path.endswith('/'):
                path = path[:-1]
                
            files = [f for f in listdir(path) if f.endswith('.bag')]
            files = rename_files(files, path)

            for file in files:
                print(f"Reading bag file {file}")
                if isdir(join(path, file.split(".")[0])):
                    print(f"Folder already exists for bag file {file}. Skipping bag reading...")
                    continue
                
                if vision:
                    mkdir(join(path, file.split(".")[0]))
                    mkdir(join(path, file.split(".")[0] + "/infra1"))
                    mkdir(join(path, file.split(".")[0] + "/infra2"))
                    mkdir(join(path, file.split(".")[0] + "/bottom"))
                    mkdir(join(path, file.split(".")[0] + "/color"))

                    write_imgs(join(path, file), join(path, file.split(".")[0]) + "/")

                write_csvs(join(path, file))
        ```
*   **tests/**
    *   **imgs.csv**:
        ```csv
        data/1c/ifo001/infra2/18624301458.jpeg
        data/1c/ifo001/infra2/33928845906.jpeg
        data/1c/ifo001/infra2/71490827346.jpeg
        data/1c/ifo001/infra2/48575590396.jpeg
        data/1c/ifo001/infra2/62084031129.jpeg
        data/1c/ifo001/infra2/405351424.jpeg
        data/1c/ifo001/infra2/59415897870.jpeg
        data/1c/ifo001/infra2/27897664094.jpeg
        data/1c/ifo001/infra2/37627923513.jpeg
        data/1c/ifo001/infra2/81233087802.jpeg
        ```
    *   **test_data.py**:
        ```python
        import unittest
        from miluv.data import DataLoader
        import numpy as np
        import os

        with open("tests/imgs.csv", "r") as f:
            img_names = [line[:-1] for line in f.readlines()]


        class TestDataLoader(unittest.TestCase):

            def setUp(self):
                self.loader = DataLoader(
                    "1c",
                    barometer=False,
                    height=False,
                    cir=False,
                )

            def test_read_csv(self):
                robot_ids = [
                    "ifo001",
                    "ifo002",
                    "ifo003",
                ]

                topics = [
                    "imu_cam",
                    "imu_px4",
                    "uwb_range",
                    "mag",
                    "uwb_passive",
                    "mocap",
                ]

                for id in robot_ids:
                    for t in topics:
                        data = self.loader.read_csv(t, id)
                        self.assertTrue(data is not None)
                        self.assertTrue(len(data) > 0)
                pass

            def test_closest_past_timestamp(self):
                timestamps = np.arange(0E9, 20E9, 1E9)

                robot_ids = [
                    "ifo001",
                    "ifo002",
                    "ifo003",
                ]

                sensor = [
                    "imu_cam",
                    "imu_px4",
                    "uwb_range",
                    "mag",
                    # "bottom",  # Uncomment when images available
                    # "color",  # Uncomment when images available
                    # "infra1",  # Uncomment when images available
                    # "infra2",  # Uncomment when images available
                ]

                for ts in timestamps:
                    for id in robot_ids:
                        for s in sensor:
                            t_closest = self.loader.closest_past_timestamp(id, s, ts)
                            self.assertTrue(t_closest is None or t_closest <= ts)
                            if t_closest is not None:
                                if s not in ["bottom", "color", "infra1", "infra2"]:
                                    idx = np.where(self.loader.data[id][s]["timestamp"]
                                                   <= ts)[0][-1]
                                    self.assertTrue(t_closest == self.loader.data[id]
                                                    [s]["timestamp"][idx])
                                else:
                                    all_imgs = os.listdir(
                                        os.path.join(self.loader.exp_dir,
                                                     self.loader.exp_name, id, s))
                                    all_imgs = [
                                        int(img.split(".")[0]) for img in all_imgs
                                    ]
                                    not_over = [t for t in all_imgs if t <= ts]
                                    t = max(not_over)
                                    self.assertTrue(t_closest == t)
                pass

            def test_data_from_timestamp(self):
                timestamps = np.arange(0E9, 20E9, 1E9)

                robot_ids = [
                    "ifo001",
                    "ifo002",
                    "ifo003",
                ]
                sensors = [
                    "imu_cam",
                    "imu_px4",
                    "uwb_range",
                    "mag",
                ]

                data = self.loader.data_from_timestamp(
                    timestamps,
                    robot_ids,
                    sensors,
                )

                self.assertTrue(data is not None)
                self.assertTrue(len(data) > 0)
                self.assertTrue(all([id in data for id in robot_ids]))
                self.assertTrue(
                    all([s in data[id] for s in sensors for id in robot_ids]))
                self.assertTrue(
                    all([
                        len(data[id][s]) <= len(timestamps) for s in sensors
                        for id in robot_ids
                    ]))
                self.assertTrue(
                    all([len(data[id][s]) > 0 for s in sensors for id in robot_ids]))
                self.assertTrue(
                    all([
                        len(data[id][s].columns) > 0 for s in sensors
                        for id in robot_ids
                    ]))
                self.assertTrue(
                    data["ifo001"]["imu_px4"].iloc[0]["timestamp"] == 996361984.0)
                self.assertTrue(data["ifo001"]["imu_px4"].iloc[0]["angular_velocity.x"]
                                == -0.0027618035674095)
                self.assertTrue(data["ifo001"]["imu_px4"].iloc[0]["angular_velocity.y"]
                                == 0.001820649835281)
                self.assertTrue(data["ifo001"]["imu_px4"].iloc[0]["angular_velocity.z"]
                                == 0.0012756492942571)
                self.assertTrue(data["ifo001"]["imu_px4"].iloc[0]
                                ["linear_acceleration.x"] == -0.4494863748550415)
                self.assertTrue(data["ifo001"]["imu_px4"].iloc[0]
                                ["linear_acceleration.y"] == 0.0494523160159599)
                self.assertTrue(data["ifo001"]["imu_px4"].iloc[0]
                                ["linear_acceleration.z"] == 9.807592391967772)

                pass

            def test_images_present(self):
                robot_ids = [
                    "ifo001",
                    "ifo002",
                    "ifo003",
                ]

                cams = [
                    "bottom",
                    "color",
                    "infra1",
                    "infra2",
                ]

                for name in img_names:
                    self.assertTrue(os.path.exists(name))

                pass

            def test_imgs_from_timestamp(self):
                timestamps = np.arange(0E9, 20E9, 1E9)

                robot_ids = [
                    "ifo001",
                    "ifo002",
                    "ifo003",
                ]

                cams = [
                    "bottom",
                    "color",
                    "infra1",
                    "infra2",
                ]

                imgs = self.loader.imgs_from_timestamps(timestamps, robot_ids, cams)

                self.assertTrue(imgs is not None)
                self.assertTrue(
                    imgs['ifo001']['bottom'].iloc[0]["timestamp"] == -13562247.0)
                self.assertTrue(
                    imgs['ifo001']['color'].iloc[0]["timestamp"] == -28261638.0)
                self.assertTrue(
                    imgs['ifo001']['infra1'].iloc[0]["timestamp"] == 972378516.0)
                self.assertTrue(
                    imgs['ifo001']['infra2'].iloc[0]["timestamp"] == 972378516.0)

                pass


        if __name__ == '__main__':
            unittest.main()
        ```
*   **uwb_ros/**
    *   **CMakeLists.txt**:
        ```cmake
        cmake_minimum_required(VERSION 3.0.2)
        project(uwb_ros)

        find_package(catkin REQUIRED COMPONENTS
          message_generation
          std_msgs
        )

        ## Generate messages in the 'msg' folder
        add_message_files(
           FILES
           RangeStamped.msg
           PassiveStamped.msg
           CirStamped.msg
        )

        ## Generate added messages and services with any dependencies listed here
        generate_messages(
          DEPENDENCIES
          std_msgs
        )

        catkin_package()

        include_directories(
            ${catkin_INCLUDE_DIRS}
        )
        ```
    *   **package.xml**:
        ```xml
        <?xml version="1.0"?>
        <package format="2">
          <name>uwb_ros</name>
          <version>0.0.0</version>
          <description>The uwb_ros package</description>

          <maintainer email="mohammed.shalaby@mail.mcgill.ca">Mohammed Shalaby</maintainer>

          <license>TODO</license>


          <build_depend>message_generation</build_depend>
          <build_export_depend>message_generation</build_export_depend>
          <exec_depend>message_runtime</exec_depend>
          <buildtool_depend>catkin</buildtool_depend>


          <export>
          </export>
        </package>
        ```
    *   **msg/**
        *   **CirStamped.msg**:
            ```
            std_msgs/Header header
            uint16 my_id
            uint16 from_id
            uint16 to_id
            uint16 idx
            int16[] cir
            ```
        *   **PassiveStamped.msg**:
            ```
            std_msgs/Header header
            uint16 my_id
            uint16 from_id
            uint16 to_id
            float64 covariance
            uint64 rx1
            uint64 rx2
            uint64 rx3
            uint64 tx1_n
            uint64 rx1_n
            uint64 tx2_n
            uint64 rx2_n
            uint64 tx3_n
            uint64 rx3_n
            float64 pr1
            float64 pr2
            float64 pr3
            float64 pr1_n
            float64 pr2_n
            ```
        *   **RangeStamped.msg**:
            ```
            std_msgs/Header header
            uint16 from_id
            uint16 to_id
            float64 covariance
            uint64 tx1
            uint64 rx1
            uint64 tx2
            uint64 rx2
            uint64 tx3
            uint64 rx3
            float64 fpp1
            float64 fpp2
            ```
