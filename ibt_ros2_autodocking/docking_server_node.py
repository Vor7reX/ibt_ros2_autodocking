# Copyright 2025 hhakim e Innobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This shebang indicates that the script should be executed with the Python 3 interpreter.
# UTF-8 encoding is useful for handling special characters in comments or strings.

# Import fundamental ROS 2 libraries.
import rclpy
import time
from rclpy.node import Node
# Import modules for managing ROS 2 actions (server and client).
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
# Import the multi-threaded executor to handle callbacks on separate threads.
from rclpy.executors import MultiThreadedExecutor
# Import QoS (Quality of Service) definitions to configure communications.
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# Import standard ROS message types.
from sensor_msgs.msg import Image, CameraInfo # Images and camera information.
from geometry_msgs.msg import PoseStamped, TransformStamped, Quaternion, Twist # Pose, transforms, quaternions, velocity commands.
from nav2_msgs.action import NavigateToPose # Nav2 navigation action.

# Import external libraries for image processing and transform management.
from cv_bridge import CvBridge # To convert ROS Image messages to OpenCV images and vice versa.
import cv2 # OpenCV library for image processing.
import cv2.aruco as aruco # OpenCV's ArUco module for marker detection.
import numpy as np # Library for numerical operations, especially with arrays.
import tf2_ros # ROS 2 TF2 for managing transforms between frames.
from tf2_ros import Buffer, TransformListener, TransformException # Specific TF2 components.
import math # Mathematical functions.
import tf_transformations # For operations on quaternions and transforms (e.g., Euler-Quaternion conversion).
import tf2_geometry_msgs # Utilities for transforming geometric messages with TF2.
from sensor_msgs.msg import BatteryState # Message for battery state.

# Import the custom 'Docking' action definition.
from ibt_ros2_interfaces.action import Docking

class DockingActionServer(Node):
    """
    Implements an action server to manage the docking/undocking process of a robot.
    The docking process is divided into multiple phases:
    - Phase 0: Navigation to a pre-docking pose using Nav2.
    - Phase 1: Initial ArUco marker detection and sample collection for an average pose.
    - Phase 2: Navigation to the averaged pre-docking pose using Nav2.
    - Phase 3: Marker detection and collection of verification samples.
    - Phase 4: Precision corrections based on ArUco for fine alignment.
    - Phase 5: Final 'strafe' movement for entering the dock.
    - Charge Verification: Battery status check to confirm successful docking.
    The undocking procedure performs a simple backward movement.
    """
    def __init__(self):
        """
        Constructor for the DockingActionServer node.
        Initializes the node, declares and reads parameters, configures subscribers,
        publishers, action clients, and the main action server.
        """
        super().__init__('docking_action_server') # Initializes the ROS 2 node with the specified name.
        
        # Declaration and reading of configurable parameters from the launch file.
        # These parameters allow tuning the docking behavior
        # without modifying the source code.

        # Parameters for Phase 0 (initial approach with Nav2).
        self.declare_parameter('phase0_goal_x', 0.13) # X coordinate of the pre-docking goal in meters.
        self.declare_parameter('phase0_goal_y', 0.87) # Y coordinate of the pre-docking goal in meters.
        self.declare_parameter('phase0_goal_yaw_degrees', 0.0) # Yaw angle of the pre-docking goal in degrees (0.0 means no rotation).
        self.declare_parameter('phase0_kp_correction', 0.5) # Proportional gain for odometric correction.
        self.declare_parameter('phase0_max_correction_vel', 0.05) # Maximum velocity for odometric corrections.
        self.declare_parameter('phase0_pose_correction_tolerance', 0.005) # Tolerance for odometric corrections.

        # Parameters for ArUco detection and pose calculation.
        self.declare_parameter('target_marker_id', 10) # ID of the target ArUco marker (e.g., on the docking station).
        self.declare_parameter('marker_size', 0.10) # Physical size of the marker in meters.
        self.declare_parameter('goal_offset_z', 0.99) # Z offset relative to the marker for the final docking pose.
        self.declare_parameter('num_samples_to_average', 50) # Number of ArUco pose samples to average.
        self.declare_parameter('num_samples_for_verification', 50) # Number of ArUco samples for the verification phase.

        # Parameters for ArUco-based corrections (Phases 2, 4).
        self.declare_parameter('kp_linear_correction', 0.5) # Proportional gain for linear corrections.
        self.declare_parameter('max_linear_vel', 0.05) # Maximum linear velocity for corrections.
        self.declare_parameter('pose_correction_tolerance', 0.005) # Tolerance for position corrections (meters).
        self.declare_parameter('kp_yaw_correction', 1.0) # Proportional gain for yaw corrections.
        self.declare_parameter('max_angular_vel', 0.25) # Maximum angular velocity for corrections.
        self.declare_parameter('yaw_correction_tolerance', 0.017) # Tolerance for yaw corrections (radians, approx. 1 degree).
        self.declare_parameter('min_angular_vel', 0.05) # Minimum angular velocity to avoid stalling at small errors.

        # Parameters for the final strafe and undocking phase.
        self.declare_parameter('strafe_speed', -0.05) # Lateral velocity for the final strafe (dock entry/exit).
        self.declare_parameter('final_strafe_offset', 0.05) # Distance offset for the final strafe.
        self.declare_parameter('undocking_distance', 0.50) # Distance to move backward during undocking.
        self.declare_parameter('charge_verification_timeout', 20.0) # New parameter: Timeout in seconds to verify charging status.
        
        # Assigning parameter values to class variables.
        self.phase0_goal_x = self.get_parameter('phase0_goal_x').get_parameter_value().double_value
        self.phase0_goal_y = self.get_parameter('phase0_goal_y').get_parameter_value().double_value
        self.phase0_goal_yaw_degrees = self.get_parameter('phase0_goal_yaw_degrees').get_parameter_value().double_value
        self.phase0_kp_correction = self.get_parameter('phase0_kp_correction').get_parameter_value().double_value
        self.phase0_max_correction_vel = self.get_parameter('phase0_max_correction_vel').get_parameter_value().double_value
        self.phase0_pose_correction_tolerance = self.get_parameter('phase0_pose_correction_tolerance').get_parameter_value().double_value
        self.target_marker_id = self.get_parameter('target_marker_id').get_parameter_value().integer_value
        self.marker_length = self.get_parameter('marker_size').get_parameter_value().double_value
        self.goal_offset_z = self.get_parameter('goal_offset_z').get_parameter_value().double_value
        self.num_samples_to_average = self.get_parameter('num_samples_to_average').get_parameter_value().integer_value
        self.num_samples_for_verification = self.get_parameter('num_samples_for_verification').get_parameter_value().integer_value
        self.kp_linear_correction = self.get_parameter('kp_linear_correction').get_parameter_value().double_value
        self.max_linear_vel = self.get_parameter('max_linear_vel').get_parameter_value().double_value
        self.pose_correction_tolerance = self.get_parameter('pose_correction_tolerance').get_parameter_value().double_value
        self.kp_yaw_correction = self.get_parameter('kp_yaw_correction').get_parameter_value().double_value
        self.max_angular_vel = self.get_parameter('max_angular_vel').get_parameter_value().double_value
        self.yaw_correction_tolerance = self.get_parameter('yaw_correction_tolerance').get_parameter_value().double_value
        self.min_angular_vel = self.get_parameter('min_angular_vel').get_parameter_value().double_value
        self.strafe_speed = self.get_parameter('strafe_speed').get_parameter_value().double_value
        self.final_strafe_offset = self.get_parameter('final_strafe_offset').get_parameter_value().double_value
        self.undocking_distance = self.get_parameter('undocking_distance').get_parameter_value().double_value
        self.charge_verification_timeout = self.get_parameter('charge_verification_timeout').get_parameter_value().double_value

        # Internal state variables to manage the action flow.
        self.state = 'IDLE' # Current state of the action server (e.g., 'IDLE', 'START_PHASE_0', 'DOCKED').
        self.goal_handle = None # Reference to the current goal received from the client.
        self.final_result = None # Final result of the action to be returned.
        self.completion_timer = None # Timer used for fixed-duration phases (e.g., strafe, undocking).
        self.charging_status = "UNKNOWN" # Battery charge status, updated by the callback.
        self.verification_start_time = None # Timestamp for the start of the charge verification phase.
        self.reset_state_variables() # Initializes state variables to their default values.
        
        # Variables for image processing and vision.
        self.camera_matrix = None # Camera intrinsic matrix, necessary for ArUco detection.
        self.dist_coeffs = None # Camera distortion coefficients.
        self.bridge = CvBridge() # CvBridge object for ROS <-> OpenCV conversion.

        # TF2 listener configuration for transforms between frames.
        self.tf_buffer = Buffer() # Buffer to store transforms.
        self.tf_listener = TransformListener(self.tf_buffer, self) # Listener to query transforms.

        # Action client to interact with Nav2.
        self._action_client_nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        # Publisher to send velocity commands to the robot.
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ArUco dictionary and detection parameters configuration.
        self.aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250) # ArUco marker dictionary.
        self.aruco_params = cv2.aruco.DetectorParameters()

        # Corner refinement method for higher precision.
        self.aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX 

        # Topic subscription configuration.
        # QoS (Quality of Service) for camera topics (often use specific settings for reliability).
        qos_profile_subscriber = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE, # Requires guaranteed message delivery.
            history=HistoryPolicy.KEEP_LAST,        # Keeps only the last message.
            depth=1,                                # Queue depth.
            durability=DurabilityPolicy.VOLATILE    # Messages are not retained by the publisher.
        )
        # Subscription to the camera info topic.
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, qos_profile_subscriber
        )
        # Subscription to the raw image topic from the camera.
        self.image_sub = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, qos_profile_subscriber
        )
        # Subscription to the battery state topic.
        self.battery_sub = self.create_subscription(
            BatteryState, '/battery_state', self.battery_state_callback, 10
        )
        
        # Main timer for the state logic loop execution.
        self.main_timer = self.create_timer(0.1, self.main_loop) # Called every 0.1 seconds.
        
        # Initialization of the 'Docking' action server.
        self._action_server = ActionServer(
            self, Docking, 'docking', # Node, action type, action name.
            execute_callback=self.execute_callback, # Callback for goal execution.
            goal_callback=self.goal_callback,       # Callback for new goal requests.
            cancel_callback=self.cancel_callback    # Callback for cancellation requests.
        )
        self.get_logger().info("Docking Action Server ready. Waiting for a goal...")

    def reset_state_variables(self):
        """
        Resets all relevant state variables at the beginning of a new action
        or after the completion/failure of a previous one, ensuring a clean state.
        """
        self.final_result = None # Clears the final result.
        self.phase0_goal_sent = False # Indicates if Phase 0 goal has been sent.
        self.target_pre_dock_pose = None # Pre-docking pose calculated from the marker.
        self.phase0_target_pose = None # Target pose for Phase 0.
        self.final_sequence_triggered = False # Flag for the final strafe sequence.
        self.verification_cycle_done = False # Flag to indicate if the ArUco verification cycle is complete.
        self.pose_samples = [] # List to store ArUco pose samples.
        
        # Cancels and resets the completion timer if active.
        if self.completion_timer:
            self.completion_timer.cancel()
        self.completion_timer = None

    def log_and_publish_feedback(self, message):
        """
        Utility function to log informative messages and simultaneously
        publish feedback to the action client.

        Args:
            message (str): The message to log and publish as feedback.
        """
        self.get_logger().info(message)
        # Publishes feedback only if a goal is active and has not been canceled/archived.
        if self.goal_handle and self.goal_handle.is_active:
            feedback_msg = Docking.Feedback()
            feedback_msg.current_status = message
            self.goal_handle.publish_feedback(feedback_msg)

    def goal_callback(self, goal_request):
        """
        Callback triggered when a client sends a new goal request.
        Evaluates whether to accept or reject the request based on the server's current state.

        Args:
            goal_request (Docking.Goal): The goal request sent by the client.

        Returns:
            rclpy.action.GoalResponse: ACCEPTS if the request is valid and the state allows it, REJECT otherwise.
        """
        command = goal_request.command
        self.get_logger().info(f"New request received with command '{command}' (Current state: {self.state})")

        # Handles the case where the robot is already docked.
        if self.state == 'DOCKED':
            if command == 'undock':
                return GoalResponse.ACCEPT # Accepts undocking only.
            else:
                self.get_logger().warn(f"Command '{command}' rejected. The robot is already docked.")
                return GoalResponse.REJECT
        
        # Rejects new requests if another procedure is already in progress.
        if self.state != 'IDLE':
            self.get_logger().warn(f"Procedure '{self.state}' already in progress. Command '{command}' rejected.")
            return GoalResponse.REJECT
        
        # Accepts valid 'dock' or 'undock' commands.
        if command in ['dock', 'undock']:
            return GoalResponse.ACCEPT
        else:
            self.get_logger().error(f"Command '{command}' not recognized. Rejected.")
            return GoalResponse.REJECT

    def cancel_callback(self, goal_handle):
        """
        Callback triggered when a client requests to cancel an active goal.
        This server always accepts cancellation requests.

        Args:
            goal_handle: The handle of the goal to be canceled.

        Returns:
            rclpy.action.CancelResponse: ACCEPTS to accept the cancellation.
        """
        # The actual cancellation logic is handled within execute_callback.
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """
        Main callback for action execution.
        Triggered when a goal has been accepted.
        Manages the action lifecycle and state transitions.

        Args:
            goal_handle: The handle of the current goal to be executed.

        Returns:
            Docking.Result: The final result of the action (success/failure and message).
        """
        self.goal_handle = goal_handle # Saves the goal handle to interact with it.
        self.reset_state_variables() # Resets state variables for the new execution.
        command = goal_handle.request.command # Gets the requested command ('dock' or 'undock').
        self.log_and_publish_feedback(f"Goal accepted, starting '{command}' procedure.")

        # Initializes the initial state based on the command.
        if command == 'dock':
            self.state = 'START_PHASE_0'
        elif command == 'undock':
            self.state = 'PERFORMING_UNDOCKING'

        # Action execution loop.
        # Remains active as long as the state is not 'IDLE' or 'DOCKED' or ROS 2 is not shut down.
        while self.state != 'IDLE' and self.state != 'DOCKED' and rclpy.ok():
            # Checks if a cancellation has been requested.
            if self.goal_handle.is_cancel_requested:
                self.cmd_vel_pub.publish(Twist()) # Stops the robot immediately.
                self.goal_handle.canceled() # Notifies the client that the action has been canceled.
                self.log_and_publish_feedback("Action canceled by client.")
                self.state = 'IDLE' # Sets the state to IDLE.
                result = Docking.Result() # Prepares the result.
                result.success = False
                result.message = "Action canceled by client."
                return result # Returns the result and ends the callback.
            
            # Pause to avoid excessive CPU usage in the waiting loop.
            time.sleep(0.2) 
        
        # Returns the final result of the action.
        # If the action has been completed (success/failure), self.final_result will be set.
        return self.final_result if self.final_result is not None else Docking.Result()

    def _finish_action(self, success, message):
        """
        Utility function to finalize the action, setting the result
        and notifying the client (succeed or abort).

        Args:
            success (bool): True if the action was completed successfully, False otherwise.
            message (str): Descriptive message of the result.
        """
        # Avoids calling the function if there is no active goal handle.
        if not self.goal_handle or not self.goal_handle.is_active:
            return
        
        result = Docking.Result() # Creates the result message.
        result.success = success
        result.message = message
        self.final_result = result # Saves the final result.

        if success:
            self.log_and_publish_feedback(f"Action completed successfully: {message}")
            self.goal_handle.succeed() # Notifies success to the client.
            # Sets the correct final state: 'DOCKED' for docking, 'IDLE' for undocking.
            if self.goal_handle.request.command == 'dock':
                self.state = 'DOCKED'
            else:
                self.state = 'IDLE'
        else:
            self.log_and_publish_feedback(f"Action failed: {message}")
            self.goal_handle.abort() # Notifies failure to the client.
            self.state = 'IDLE' # In case of failure, the robot always returns to IDLE state.

    def main_loop(self):
        """
        The main loop of the node, called periodically by a timer.
        Manages transitions between the different phases of the docking/undocking procedure
        based on the current server state.
        """
        # If the server is in 'IDLE' or 'DOCKED' state, there is no active logic to execute in this loop.
        if self.state in ['IDLE', 'DOCKED']: return
        
        # Logic for transitioning between docking/undocking phases.
        if self.state == 'START_PHASE_0' and not self.phase0_goal_sent:
            # Starts Phase 0 (Navigation to pre-docking pose with Nav2).
            self.start_phase_zero()
        elif self.state in ['PHASE0_CORRECTING_X', 'PHASE0_CORRECTING_Y']:
            # Executes the odometric pose corrections of Phase 0.
            self.phase_zero_correction_loop()
        elif self.state in ['CORRECTING_X_POSE', 'CORRECTING_Y_POSE', 'CORRECTING_YAW']:
            # Executes ArUco-based pose and orientation corrections.
            self.correction_logic()
        elif self.state == 'VERIFYING_CHARGE':
            # Checks the battery charge status.
            self.verify_charge_status()
        elif self.state == 'PERFORMING_UNDOCKING':
            # Initial state for the undocking procedure.
            self.state = 'UNDOCKING_IN_PROGRESS' # Immediate transition to start undocking.
            self.start_undocking_procedure()
    
    def battery_state_callback(self, msg):
        """
        Callback triggered when a battery state message is received.
        Updates the internal 'charging_status' variable based on the current drawn/supplied.

        Args:
            msg (sensor_msgs.msg.BatteryState): The battery state message.
        """
        current = msg.current
        CHARGE_THRESHOLD = 0.5    # Minimum current to be considered charging (Amperes).
        DISCHARGE_THRESHOLD = -0.5 # Maximum current to be considered discharging (negative Amperes).
        
        if current > CHARGE_THRESHOLD:
            self.charging_status = "Is charging"
        elif current < DISCHARGE_THRESHOLD:
            self.charging_status = "Is discharging"
        else:
            self.charging_status = "Is not charging" # Neutral, neither charging nor significantly discharging.
    
    def verify_charge_status(self):
        """
        Checks if the robot is actually charging after docking.
        Includes a timeout to avoid waiting indefinitely.
        """
        # Calculates the elapsed time since the start of verification.
        elapsed_time = (self.get_clock().now() - self.verification_start_time).nanoseconds / 1e9
        
        # If the timeout is exceeded, the action fails.
        if elapsed_time > self.charge_verification_timeout:
            self._finish_action(False, f"Docking failed: the robot is not charging after {self.charge_verification_timeout} seconds.")
            return

        # Logs and publishes feedback on the current verification status.
        self.log_and_publish_feedback(f"Verifying charge status... (Current: {self.charging_status})")
        
        # If the robot is charging, the docking action is considered successful.
        if self.charging_status == "Is charging":
            self._finish_action(True, "Docking completed and robot is charging.")

    def start_phase_zero(self):
        """
        Starts Phase 0: sends a navigation goal to Nav2 to bring the robot
        close to the docking area, based on predefined coordinates.
        """
        self.phase0_goal_sent = True # Marks that the goal has been sent.
        
        # Creates the PoseStamped message for the Nav2 goal.
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = 'map' # The goal is in the 'map' frame.
        goal_pose.pose.position.x = self.phase0_goal_x
        goal_pose.pose.position.y = self.phase0_goal_y
         # Converte l'angolo di yaw da gradi a radianti
        yaw_rad = math.radians(self.phase0_goal_yaw_degrees)
        # Converte l'angolo di yaw in un quaternione (pitch e roll a 0)
        q = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw_rad)
        # Assegna il quaternione al goal_pose
        goal_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.phase0_target_pose = goal_pose # Saves the target for subsequent corrections.
        
        self.log_and_publish_feedback(f"PHASE 0: Sending pre-docking goal to Nav2: X={self.phase0_goal_x}, Y={self.phase0_goal_y}")
        self.send_goal_to_nav2(goal_pose) # Sends the goal to Nav2.

    def phase_zero_correction_loop(self):
        """
        Performs precision pose corrections in Phase 0, after Nav2
        has completed its coarse navigation.
        Uses odometric feedback to align the robot along the X and Y axes.
        """
        if self.phase0_target_pose is None: return # Does not proceed if the target is not set.
        
        try:
            # Gets the current transform of the robot ('base_link') relative to the 'map' frame.
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            current_x, current_y = trans.transform.translation.x, trans.transform.translation.y
        except TransformException as ex:
            # Handles errors in retrieving the transform.
            self.log_and_publish_feedback(f"PHASE 0: Could not get robot pose: {ex}"); return
        
        twist_msg = Twist() # Initializes the velocity message to zero.

        if self.state == 'PHASE0_CORRECTING_X':
            error_x = self.phase0_target_pose.pose.position.x - current_x # Calculates the error in X.
            self.log_and_publish_feedback(f"PHASE 0 - X Correction: Error={error_x:.4f}m")
            
            # If the error in X is within tolerance, proceeds to Y correction.
            if abs(error_x) < self.phase0_pose_correction_tolerance:
                self.log_and_publish_feedback("PHASE 0: X correction completed.")
                self.state = 'PHASE0_CORRECTING_Y'
            else:
                # Applies a proportional correction, limited by maximum velocity.
                twist_msg.linear.x = np.clip(
                    self.phase0_kp_correction * error_x, 
                    -self.phase0_max_correction_vel, 
                    self.phase0_max_correction_vel
                )
        elif self.state == 'PHASE0_CORRECTING_Y':
            error_y = self.phase0_target_pose.pose.position.y - current_y # Calculates the error in Y.
            self.log_and_publish_feedback(f"PHASE 0 - Y Correction: Error={error_y:.4f}m")
            
            # If the error in Y is within tolerance, Phase 0 is complete.
            if abs(error_y) < self.phase0_pose_correction_tolerance:
                self.log_and_publish_feedback("PHASE 0: Y correction completed.")
                self.log_and_publish_feedback("TRANSITION: Starting PHASE 1 - ArUco marker search.")
                self.state = 'SEARCHING_INITIAL_GOAL' # Transitions to the ArUco search phase.
                self.cmd_vel_pub.publish(Twist()) # Stops the robot.
                return
            else:
                # Applies a proportional correction, limited by maximum velocity.
                twist_msg.linear.y = np.clip(
                    self.phase0_kp_correction * error_y, 
                    -self.phase0_max_correction_vel, 
                    self.phase0_max_correction_vel
                )
        self.cmd_vel_pub.publish(twist_msg) # Publishes the velocity command.

    def camera_info_callback(self, msg):
        """
        Callback triggered when camera calibration information is received.
        Saves the intrinsic matrix and distortion coefficients, essential for ArUco detection.

        Args:
            msg (sensor_msgs.msg.CameraInfo): The message with camera calibration parameters.
        """
        # Camera parameters are only needed once.
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape([3, 3]) # Intrinsic (K) matrix.
            self.dist_coeffs = np.array(msg.d) # Distortion coefficients.
            self.log_and_publish_feedback("Camera calibration parameters received.")
            
    def image_callback(self, msg):
        """
        Callback triggered when a new image is received from the camera.
        If the server is in a state that requires vision, it detects ArUco markers
        and handles sample collection or activation of the final phase.

        Args:
            msg (sensor_msgs.msg.Image): The ROS image message.
        """
        # Processes the image only if the server is in an ArUco-relevant state.
        if self.state not in ['SEARCHING_INITIAL_GOAL', 'COLLECTING_GOAL_SAMPLES', 'AWAITING_FINAL_ARUCO_MEASUREMENT', 'COLLECTING_VERIFICATION_SAMPLES']:
            return
        
        if self.camera_matrix is None: return # Requires camera calibration parameters.

        # Converts the ROS image message to an OpenCV image.
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) # Converts to grayscale for detection.

        # Detects ArUco markers in the image.
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        # If markers are detected and the target marker is present.
        if ids is not None and self.target_marker_id in ids:
            # Estimates the pose (rotation and translation) of the markers relative to the camera.
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            for i, marker_id in enumerate(ids):
                if marker_id[0] == self.target_marker_id:
                    if self.state == 'SEARCHING_INITIAL_GOAL':
                        # If the marker is found for the first time.
                        self.log_and_publish_feedback("PHASE 1: Marker found, starting sample collection.")
                        self.state = 'COLLECTING_GOAL_SAMPLES' # Transitions to the sample collection phase.
                    elif self.state in ['COLLECTING_GOAL_SAMPLES', 'COLLECTING_VERIFICATION_SAMPLES']:
                        # Collects a pose sample of the marker.
                        self.collect_goal_sample(msg.header, tvecs[i][0], rvecs[i][0])
                    elif self.state == 'AWAITING_FINAL_ARUCO_MEASUREMENT' and not self.final_sequence_triggered:
                        # If it's the waiting phase for the final measurement before strafing.
                        self.final_sequence_triggered = True # Activates the final sequence.
                        self.log_and_publish_feedback("PHASE 5: Final measurement obtained, starting translation.")
                        self.state = 'PERFORMING_FINAL_STRAFE' # Transitions to the strafe phase.
                        self.start_final_strafe(tvecs[i][0]) # Starts the final strafe.
                    break # Once target_marker_id is found, exits the loop.
    
    def collect_goal_sample(self, header, tvec, rvec):
        """
        Collects a single ArUco marker pose sample, transforms it to the 'map' frame,
        and calculates the desired pre-docking pose, adding it to the list of samples.

        Args:
            header (std_msgs.msg.Header): Header of the image message (contains timestamp and frame_id).
            tvec (numpy.ndarray): Translation vector of the marker relative to the camera.
            rvec (numpy.ndarray): Rotation vector of the marker relative to the camera (Rodrigues format).
        """
        pose_in_camera = PoseStamped() # Creates a pose message for the marker in the camera frame.
        pose_in_camera.header = header
        pose_in_camera.header.frame_id = "camera_color_optical_frame" # Camera frame.
        # Assigns translation components.
        pose_in_camera.pose.position.x, pose_in_camera.pose.position.y, pose_in_camera.pose.position.z = float(tvec[0]), float(tvec[1]), float(tvec[2])
        # Converts the Rodrigues vector to a quaternion for orientation.
        qx, qy, qz, qw = self.rodrigues_to_quaternion(rvec)
        pose_in_camera.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        try:
            # Gets the transform from the camera frame to the 'map' frame.
            transform = self.tf_buffer.lookup_transform(
                'map', pose_in_camera.header.frame_id, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)
            )
            # Transforms the marker's pose from the camera frame to the 'map' frame.
            marker_in_map = tf2_geometry_msgs.do_transform_pose_stamped(pose_in_camera, transform)
            
            # Calculates the desired pre-docking pose based on the marker's pose.
            pre_dock_goal = self.calculate_pre_dock_goal(marker_in_map)
            
            if pre_dock_goal:
                self.pose_samples.append(pre_dock_goal) # Adds the calculated goal to the list of samples.
                
                # Determines the target number of samples for the current phase.
                target_samples = self.num_samples_for_verification if self.state == 'COLLECTING_VERIFICATION_SAMPLES' else self.num_samples_to_average
                
                self.log_and_publish_feedback(f"Collected sample {len(self.pose_samples)}/{target_samples} (State: {self.state})")
                
                # If the required number of samples has been reached.
                if len(self.pose_samples) >= target_samples:
                    if self.state == 'COLLECTING_GOAL_SAMPLES':
                        # If it's the initial collection phase, proceeds to Nav2 navigation.
                        self.log_and_publish_feedback("PHASE 1: Initial sample collection completed.")
                        self.state = 'MOVING_TO_PRE_DOCK_GOAL'
                        self.calculate_average_and_send_goal() # Calculates the average and sends the goal to Nav2.
                    elif self.state == 'COLLECTING_VERIFICATION_SAMPLES':
                        # If it's the collection phase for verification, processes the samples.
                        self.log_and_publish_feedback("PHASE 3: Verification sample collection completed.")
                        self.process_verification_samples()
        except TransformException as ex:
            # Handles errors in TF2 transform.
            self.log_and_publish_feedback(f"ATTENTION: Sample discarded (TF): {ex}")
    
    def calculate_average_and_send_goal(self):
        """
        Calculates the average pose from all collected samples and sends it to Nav2
        as a goal for precise navigation to the pre-docking pose.
        """
        if not self.pose_samples:
            self._finish_action(False, "No samples collected."); self.state='IDLE'; return
        
        # Calculates the average of X, Y, Z positions.
        avg_x = sum(p.pose.position.x for p in self.pose_samples) / len(self.pose_samples)
        avg_y = sum(p.pose.position.y for p in self.pose_samples) / len(self.pose_samples)
        avg_z = sum(p.pose.position.z for p in self.pose_samples) / len(self.pose_samples)
        
        # Averages quaternions. This is a simplified method (sum and normalization)
        # that works well for small variations. For extreme precision, methods like SVD would be used.
        q_sum = np.array([0.0, 0.0, 0.0, 0.0])
        for p in self.pose_samples:
            q_sum += [p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w]
        q_avg_norm = q_sum / np.linalg.norm(q_sum) # Normalizes the average quaternion.

        averaged_goal = PoseStamped() # Creates the averaged pose message.
        averaged_goal.header.frame_id = 'map'
        averaged_goal.header.stamp = self.get_clock().now().to_msg()
        averaged_goal.pose.position.x, averaged_goal.pose.position.y, averaged_goal.pose.position.z = avg_x, avg_y, avg_z
        averaged_goal.pose.orientation = Quaternion(x=q_avg_norm[0], y=q_avg_norm[1], z=q_avg_norm[2], w=q_avg_norm[3])
        
        self.target_pre_dock_pose = averaged_goal # Saves the average pose as the target for subsequent corrections.
        
        # Calculates the yaw angle from the quaternion for logging.
        (_, _, avg_yaw) = tf_transformations.euler_from_quaternion(q_avg_norm)
        
        self.log_and_publish_feedback(f"PHASE 2: Sending averaged goal (ArUco): X={avg_x:.3f}, Y={avg_y:.3f}, Yaw={math.degrees(avg_yaw):.2f} degrees")
        self.send_goal_to_nav2(averaged_goal) # Sends the goal to Nav2.

    def correction_logic(self):
        """
        Performs precise robot corrections based on the ArUco marker's pose.
        This loop corrects the X, Y position and Yaw orientation
        relative to the averaged dock pose.
        """
        if self.target_pre_dock_pose is None: return # Does not proceed if the target is not set.
        
        try:
            # Gets the current transform of the robot ('base_link') relative to the 'map' frame.
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.2))
        except TransformException as ex:
            self.log_and_publish_feedback(f"ATTENTION: ArUco Correction: Could not get pose: {ex}"); return
        
        twist_msg = Twist() # Initializes the velocity message.

        if self.state == 'CORRECTING_X_POSE':
            error_x = self.target_pre_dock_pose.pose.position.x - trans.transform.translation.x # Calculates X error.
            self.log_and_publish_feedback(f"ArUco X Correction: Error={error_x:.4f}m")
            
            if abs(error_x) < self.pose_correction_tolerance:
                self.log_and_publish_feedback("ArUco X correction completed.")
                self.state = 'CORRECTING_Y_POSE' # Proceeds to Y correction.
            else:
                # Applies proportional correction in X, limited.
                twist_msg.linear.x = np.clip(self.kp_linear_correction * error_x, -self.max_linear_vel, self.max_linear_vel)
        
        elif self.state == 'CORRECTING_Y_POSE':
            error_y = self.target_pre_dock_pose.pose.position.y - trans.transform.translation.y # Calculates Y error.
            self.log_and_publish_feedback(f"ArUco Y Correction: Error={error_y:.4f}m")
            
            if abs(error_y) < self.pose_correction_tolerance:
                self.log_and_publish_feedback("ArUco Y correction completed.")
                self.state = 'CORRECTING_YAW' # Proceeds to Yaw correction.
            else:
                # Applies proportional correction in Y, limited.
                twist_msg.linear.y = np.clip(self.kp_linear_correction * error_y, -self.max_linear_vel, self.max_linear_vel)
        
        elif self.state == 'CORRECTING_YAW':
            current_q = trans.transform.rotation
            # Extracts the current yaw angle of the robot.
            (_, _, current_yaw) = tf_transformations.euler_from_quaternion([current_q.x, current_q.y, current_q.z, current_q.w])
            
            target_q = self.target_pre_dock_pose.pose.orientation
            # Extracts the target yaw angle from the dock.
            (_, _, target_yaw) = tf_transformations.euler_from_quaternion([target_q.x, target_q.y, target_q.z, target_q.w])
            
            # Calculates the yaw error, handling 2*pi wrapping.
            error_yaw = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
            self.log_and_publish_feedback(f"ArUco Yaw Correction: Error={math.degrees(error_yaw):.2f} degrees")
            
            if abs(error_yaw) < self.yaw_correction_tolerance:
                # If the yaw error is within tolerance.
                if not self.verification_cycle_done:
                    # If it's the first time ArUco corrections are completed, starts the verification phase.
                    self.log_and_publish_feedback("PHASE 2: First ArUco correction completed. Starting PHASE 3: Verification.")
                    self.setup_verification_cycle()
                else:
                    # If it's the second correction cycle (post-verification), proceeds to the final phase.
                    self.log_and_publish_feedback("PHASE 4: Final post-verification correction completed. Starting PHASE 5: Final Strafe.")
                    self.state = 'AWAITING_FINAL_ARUCO_MEASUREMENT' # Waits for the last ArUco measurement for strafe.
            else:
                # Applies proportional yaw correction, limited and with a minimum velocity.
                angular_vel = self.kp_yaw_correction * error_yaw
                if 0 < abs(angular_vel) < self.min_angular_vel:
                    angular_vel = self.min_angular_vel * np.sign(angular_vel) # Applies minimum velocity to overcome friction.
                twist_msg.angular.z = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        self.cmd_vel_pub.publish(twist_msg) # Publishes the velocity command.

    def setup_verification_cycle(self):
        """
        Prepares the server for the pose verification phase (Phase 3).
        Resets the sample list and sets the state for new collection.
        """
        self.verification_cycle_done = True # Indicates that the first correction cycle is finished.
        self.pose_samples = [] # Clears previous samples.
        self.log_and_publish_feedback(f"PHASE 3: Collecting {self.num_samples_for_verification} new samples...")
        self.state = 'COLLECTING_VERIFICATION_SAMPLES' # Changes state to collect verification samples.
    
    def process_verification_samples(self):
        """
        Processes the samples collected in the verification phase (Phase 3).
        Calculates a new average pose and sets it as the target for the second correction cycle.
        """
        if not self.pose_samples:
            self._finish_action(False, "No verification samples collected."); self.state='IDLE'; return
        
        # Calculates the average of positions and orientation from verification samples.
        avg_x = sum(p.pose.position.x for p in self.pose_samples) / len(self.pose_samples)
        avg_y = sum(p.pose.position.y for p in self.pose_samples) / len(self.pose_samples)
        avg_z = sum(p.pose.position.z for p in self.pose_samples) / len(self.pose_samples)
        
        q_sum = np.array([0.0, 0.0, 0.0, 0.0])
        for p in self.pose_samples:
            q_sum += [p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w]
        q_avg_norm = q_sum / np.linalg.norm(q_sum)

        # Updates the target pose with the averaged verification values.
        self.target_pre_dock_pose.pose.position.x = avg_x
        self.target_pre_dock_pose.pose.position.y = avg_y
        self.target_pre_dock_pose.pose.position.z = avg_z
        self.target_pre_dock_pose.pose.orientation = Quaternion(x=q_avg_norm[0], y=q_avg_norm[1], z=q_avg_norm[2], w=q_avg_norm[3])
        
        self.pose_samples = [] # Clears the sample list for next use.
        
        self.log_and_publish_feedback("Starting PHASE 4: Second ArUco correction cycle.")
        self.state = 'CORRECTING_X_POSE' # Returns to the correction phase for final refinement.

    def calculate_pre_dock_goal(self, marker_pose_in_map: PoseStamped):
        """
        Calculates the desired final pose for the robot relative to the ArUco marker's pose
        in the 'map' frame. Applies an offset to position the robot correctly for docking.

        Args:
            marker_pose_in_map (PoseStamped): The pose of the ArUco marker in the 'map' frame.

        Returns:
            PoseStamped: The desired target pose for the robot in the 'map' frame.
        """
        final_goal = PoseStamped()
        final_goal.header.frame_id = 'map'
        final_goal.header.stamp = self.get_clock().now().to_msg()
        
        # Defines the Z-offset vector relative to the marker.
        offset_vector = np.array([0.0, 0.0, self.goal_offset_z])
        
        # Extracts the quaternion of the marker's orientation.
        marker_q_list = [marker_pose_in_map.pose.orientation.x, marker_pose_in_map.pose.orientation.y, marker_pose_in_map.pose.orientation.z, marker_pose_in_map.pose.orientation.w]
        
        # Creates a rotation matrix from the marker's quaternion.
        marker_rotation_matrix = tf_transformations.quaternion_matrix(marker_q_list)[:3, :3]
        
        # Rotates the offset vector to align with the marker's orientation.
        rotated_offset = marker_rotation_matrix.dot(offset_vector)
        
        # Calculates the final position by adding the rotated offset to the marker's position.
        final_goal.pose.position.x = marker_pose_in_map.pose.position.x + rotated_offset[0]
        final_goal.pose.position.y = marker_pose_in_map.pose.position.y + rotated_offset[1]
        final_goal.pose.position.z = marker_pose_in_map.pose.position.z + rotated_offset[2]
        
        # Maintains the marker's Yaw orientation for the robot.
        (_, _, yaw) = tf_transformations.euler_from_quaternion(marker_q_list)
        goal_q = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw) # Quaternion with yaw only.
        final_goal.pose.orientation = Quaternion(x=goal_q[0], y=goal_q[1], z=goal_q[2], w=goal_q[3])
        
        return final_goal

    def start_final_strafe(self, tvec):
        """
        Starts the final strafe (lateral movement) phase for docking.
        Calculates the distance to travel and the duration of the movement.

        Args:
            tvec (numpy.ndarray): Translation vector of the marker relative to the camera (including Z distance).
        """
        # The lateral distance to travel is the marker's Z distance minus a final offset.
        lateral_distance = abs(tvec[2]) - self.final_strafe_offset
        
        if lateral_distance <= 0:
            self.log_and_publish_feedback(f"Robot already at target distance. Skipping strafe.")
            self.finish_procedure() # If already close enough, skips strafe and proceeds to verification.
            return
        
        if self.strafe_speed == 0:
            self._finish_action(False, "strafe_speed is 0."); self.state='IDLE'; return
        
        # Calculates the duration of the strafe based on distance and speed.
        duration = lateral_distance / abs(self.strafe_speed)
        
        self.log_and_publish_feedback(f"PHASE 5: Final Strafe. Duration: {duration:.3f}s.")
        
        twist_msg = Twist()
        twist_msg.linear.y = self.strafe_speed # Sets the lateral velocity (robot's Y-axis).
        self.cmd_vel_pub.publish(twist_msg) # Publishes the velocity command.
        
        # Creates a timer that will call finish_procedure at the end of the strafe.
        self.completion_timer = self.create_timer(duration, self.finish_procedure)

    def start_undocking_procedure(self):
        """
        Starts the undocking procedure: a backward movement of the robot
        to move away from the docking station.
        """
        self.log_and_publish_feedback("UNDOCKING PHASE: Starting undocking procedure.")
        
        undocking_speed = -self.strafe_speed # Undocking speed is the opposite of strafe_speed.
        
        try:
            # Calculates the duration of undocking.
            duration = self.undocking_distance / undocking_speed
        except ZeroDivisionError:
            self._finish_action(False, "Undocking speed is zero."); self.state='IDLE'; return
        
        self.log_and_publish_feedback(f"UNDOCKING PHASE: Performing undocking of {self.undocking_distance:.2f}m.")
        
        twist_msg = Twist()
        twist_msg.linear.y = undocking_speed # Moves the robot backward (along the robot's Y-axis).
        self.cmd_vel_pub.publish(twist_msg) # Publishes the velocity command.
        
        # Creates a timer that will call finish_undocking_procedure at the end of the movement.
        self.completion_timer = self.create_timer(duration, self.finish_undocking_procedure)
        
    def finish_procedure(self):
        """
        This function is called at the end of the 'strafe' phase (Phase 5).
        Instead of ending the docking action, it starts the charge verification phase.
        """
        if self.completion_timer:
            self.completion_timer.cancel() # Cancels the completion timer.
        
        self.log_and_publish_feedback("PHASE 5 (Strafe) completed. Starting charge verification.")
        self.cmd_vel_pub.publish(Twist()) # Stops the robot.
        
        # MODIFICATION: Starts the charge verification phase.
        self.state = 'VERIFYING_CHARGE'
        self.verification_start_time = self.get_clock().now() # Sets the verification start timestamp.

    def finish_undocking_procedure(self):
        """
        Finalizes the undocking procedure.
        Called at the end of the backward movement.
        """
        if self.completion_timer:
            self.completion_timer.cancel() # Cancels the completion timer.
        self.cmd_vel_pub.publish(Twist()) # Stops the robot.
        self._finish_action(True, "Undocking completed successfully.") # Notifies success to the client.
        self.state = 'IDLE' # Returns to IDLE state.

    def send_goal_to_nav2(self, goal_pose):
        """
        Sends a navigation goal to the Nav2 client.

        Args:
            goal_pose (geometry_msgs.msg.PoseStamped): The target pose for Nav2.
        """
        # Waits for the Nav2 server to be available.
        if not self._action_client_nav.wait_for_server(timeout_sec=5.0):
            self._finish_action(False, 'Nav2 Action server not available.'); self.state='IDLE'; return
        
        self.log_and_publish_feedback('Nav2 Action server found. Sending goal...')
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose # Sets the target pose in the goal message.
        
        # Sends the goal to Nav2 asynchronously.
        self._send_goal_future = self._action_client_nav.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback) # Attaches the callback for goal response.

    def goal_response_callback(self, future):
        """
        Callback triggered when Nav2 responds to a goal submission (acceptance/rejection).

        Args:
            future: A Future object containing the Nav2 goal handle.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._finish_action(False, 'Goal rejected by Nav2.'); self.state='IDLE'; return
        
        self.log_and_publish_feedback('Goal accepted by Nav2.')
        
        # Gets the final result of the Nav2 action asynchronously.
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.nav2_result_callback) # Attaches the callback for the result.

    def nav2_result_callback(self, future):
        """
        Callback triggered when Nav2 provides the final navigation result.
        Manages the transition to the next phase or action failure.

        Args:
            future: A Future object containing the Nav2 action result.
        """
        try:
            result = future.result()
            status = result.status # Gets the Nav2 result status.
        except Exception as e:
            self._finish_action(False, f"Exception in Nav2 result: {e}"); self.state='IDLE'; return
        
        # Status code 4 in Nav2.action.Result (or its internal interface) indicates SUCCEEDED.
        if status != 4: 
            self._finish_action(False, f'Navigation failed with status: {status}'); self.state='IDLE'; return
        
        # Manages state transitions based on the current phase.
        if self.state == 'START_PHASE_0':
            self.log_and_publish_feedback('PHASE 0 (Nav2) completed. Starting odometric correction.')
            self.state = 'PHASE0_CORRECTING_X' # Transitions to Phase 0 odometric correction.
        elif self.state == 'MOVING_TO_PRE_DOCK_GOAL':
            self.log_and_publish_feedback('PHASE 2 (Nav2) completed. Starting first ArUco correction.')
            self.state = 'CORRECTING_X_POSE' # Transitions to the first ArUco correction.

    def rodrigues_to_quaternion(self, rvec):
        """
        Converts a Rodrigues rotation vector (used by OpenCV) to a quaternion.

        Args:
            rvec (numpy.ndarray): The Rodrigues rotation vector (3x1 or 1x3).

        Returns:
            tuple: A tuple (qx, qy, qz, qw) representing the quaternion.
        """
        rotation_matrix, _ = cv2.Rodrigues(rvec) # Converts Rodrigues to rotation matrix.
        
        # Implementation of the conversion from rotation matrix to quaternion.
        # This code block is a standard formula for conversion.
        trace = np.trace(rotation_matrix)
        if trace > 0.0:
            s = 0.5 / math.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
        elif (rotation_matrix[0, 0] > rotation_matrix[1, 1]) and (rotation_matrix[0, 0] > rotation_matrix[2, 2]):
            s = 2.0 * math.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            s = 2.0 * math.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            qz = 0.25 * s
        return qx, qy, qz, qw

def main(args=None):
    """
    Main entry point of the program.
    Initializes ROS 2, creates the action server, and starts the multi-threaded executor.
    Also handles clean shutdown in case of interruption.
    """
    rclpy.init(args=args) # Initializes the ROS 2 context.
    docking_action_server = DockingActionServer() # Creates an instance of the action server.
    
    # Creates a multi-threaded executor to allow callbacks (timers, subscriptions, actions)
    # to be processed in parallel, improving node responsiveness.
    executor = MultiThreadedExecutor()
    executor.add_node(docking_action_server) # Adds the node to the executor.
    
    try:
        executor.spin() # Starts the executor's spin loop, blocking execution here.
    except KeyboardInterrupt:
        # Handles keyboard interruption (Ctrl+C).
        docking_action_server.get_logger().info('Keyboard interrupt (Ctrl+C) detected.')
    finally:
        # Block executed always, regardless of whether an exception occurred or not.
        docking_action_server.get_logger().info('EMERGENCY SHUTDOWN: Sending stop command to robot.')
        stop_msg = Twist()
        docking_action_server.cmd_vel_pub.publish(stop_msg) # Sends a zero velocity command to stop the robot.
        
        executor.shutdown() # Shuts down the executor.
        docking_action_server.destroy_node() # Destroys the node and releases its resources.
        
        # Ensures that ROS 2 is shut down if it's still active.
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main() # Invokes the main function when the script is executed.