from setuptools import find_packages, setup
import os
from glob import glob

# The name of the ROS 2 package, which must match the directory name.
package_name = 'ibt_ros2_autodocking'

setup(
    name=package_name, # The package name as registered in the system.
    version='0.0.0',   # The current version of the package. Usually starts at 0.0.0 for packages in development.
    
    # Automatically finds all Python submodules within the package directory.
    # Excludes the 'test' directory to prevent test files from being included in the installed package.
    packages=find_packages(exclude=['test']),
    
    # Specifies data files that should be installed with the package.
    data_files=[
        # Registers the package in the ament_index resource index, necessary for ROS 2 package discovery.
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*'))),
        
        # Includes the package.xml file, which contains crucial metadata about the package (description, dependencies, etc.).
        ('share/' + package_name, ['package.xml']),
    ],
    
    # Defines the Python dependencies that must be installed for the package to function.
    # 'setuptools' is a common dependency for Python packages.
    install_requires=[
        'setuptools',
        'rclpy',             # Necessary dependency for any ROS 2 node in Python.
        'rclpy_action',      # Required for action clients and servers.
        'numpy',             # Used for numerical calculations (e.g., in docking_action_server).
        'opencv-python',     # For OpenCV functionalities, including ArUco.
        'opencv-contrib-python', # Often needed for ArUco, which is in the 'contrib' module.
        'tf-transformations',# For conversions of quaternions and transformation matrices.
        # Make sure to add any other Python libraries your code directly imports here
        # that are not already provided by a ROS 2 package (e.g., scipy, matplotlib, etc.).
    ],
    
    # 'zip_safe=True' indicates that the package can be installed as a .zip file.
    # For ROS 2 packages, it's often set to False if there are resources that need to be accessed as actual file systems.
    # However, for simple Python scripts, 'True' is acceptable.
    zip_safe=True,
    
    # Package maintainer information.
    maintainer='hhakim', # The maintainer's name.
    maintainer_email='hhakim815@gmail.com', # Your email.
    
    # Brief description of the package. It's important that it's concise but clear.
    description='ROS 2 package for automated robot docking and undocking using Nav2 and ArUco markers for precise alignment and battery charge verification.',
    
    # Declaration of the package license. It is fundamental to specify a license
    # I used Apache 2.0 
    license='Apache-2.0',
    
    # Defines dependencies required only for running tests.
    tests_require=['pytest'],
    
    # This is a key point for ROS 2 packages containing Python executables.
    # 'console_scripts' creates executable scripts in the system's PATH, allowing nodes to be launched directly.
    entry_points={
        'console_scripts': [
            # Associates the script name 'docking_client_node' with the 'main' function in the module.
            # The module is 'ibt_ros2_autodocking.docking_client_node' (package_name.python_file_name_without_ext).
            'docking_client_node = ibt_ros2_autodocking.docking_client_node:main',
            
            # Associates the script name 'docking_server_node' with the 'main' function in the module.
            # This is the entry point for your docking server.
            'docking_server_node = ibt_ros2_autodocking.docking_server_node:main',
        ],
    },
)