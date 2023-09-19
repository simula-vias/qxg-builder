#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 00:38:39 2023

@author: nassim
"""

from scipy.spatial.transform import Rotation as R
import pandas as pd

def compute_angular_velocity(ego_poses):
    angular_velocities = []
    
    for i in range(1, len(ego_poses)):
        # Convert quaternions to Euler angles
        euler_angles_i = R.from_quat(ego_poses[i]['orientation']).as_euler('zyx', degrees=True)
        euler_angles_i_minus_1 = R.from_quat(ego_poses[i-1]['orientation']).as_euler('zyx', degrees=True)
        
        # Calculate the orientation difference
        delta_orientation = np.array(euler_angles_i) - np.array(euler_angles_i_minus_1)
        
        # Calculate the time difference
        delta_time = (ego_poses[i]['timestamp'] - ego_poses[i-1]['timestamp']) * 1e-6#1e-6 because timestamp is in micro seconds.
        
        # Calculate the angular velocity
        angular_velocity = delta_orientation / delta_time
        
        angular_velocities.append(angular_velocity)
    
    return angular_velocities


def compute_angular_velocity_pair(object1_positions, object2_positions, timestamps):
    # Convert positions to numpy arrays for easier computation
    object1_positions = np.array(object1_positions)
    object2_positions = np.array(object2_positions)

    # Compute the relative vector between the two objects
    relative_vector = object2_positions - object1_positions

    # Compute the time differences between consecutive timestamps
    time_diff = np.diff(timestamps)

    # Compute the change in angle over time
    change_in_angle = np.arctan2(relative_vector[:, 1], relative_vector[:, 0])

    # Compute the angular velocity by dividing the change in angle by the corresponding time differences
    angular_velocity = change_in_angle / time_diff

    return angular_velocity

def compute_acceleration(velocities, ego_poses):
    accelerations = []
    
    for i in range(1, len(velocities)):
        # Calculate the velocity difference
        delta_velocity = velocities[i] - velocities[i-1]
        
        # Calculate the time difference
        delta_time = (ego_poses[i+1]['timestamp'] - ego_poses[i]['timestamp']) * 1e-6#1e-6 because timestamp is in micro seconds.
        
        # Calculate the acceleration
        acceleration = delta_velocity / delta_time
        
        accelerations.append(acceleration)
    
    return accelerations
import numpy as np

def compute_velocity(ego_poses):
    velocities = []
    
    for i in range(1, len(ego_poses)):
        # Calculate the position difference
        delta_position = np.array(ego_poses[i]['position']) - np.array(ego_poses[i-1]['position'])
        
        # Calculate the time difference
        delta_time = (ego_poses[i]['timestamp'] - ego_poses[i-1]['timestamp']) * 1e-6 #1e-6 because timestamp is in micro seconds.
        
        # Calculate the velocity
        velocity = delta_position / delta_time
        
        velocities.append(velocity)
    
    return velocities
def compute_velocity_object(ego_poses,timestamps):
    velocities = []
    
    for i in range(1, len(ego_poses)):
        # Calculate the position difference
        delta_position = np.array(ego_poses[i]) - np.array(ego_poses[i-1])
        
        # Calculate the time difference
        delta_time = (timestamps[i]- timestamps[i-1]) * 1e-6 #1e-6 to transform to seconds because timestamp is in micro seconds.
        
        # Calculate the velocity
        velocity = delta_position / delta_time
        
        velocities.append(velocity)
    
    return velocities

import numpy as np
def getEgoPoses(nusc,scene,sensor = 'LIDAR_TOP'):
    ego_poses = {}
    
    
    
    first_sample_token = scene['first_sample_token']
    current_sample = nusc.get('sample', first_sample_token)
    
    while current_sample is not None:
        # Get ego_pose for the current sample
        lidar = nusc.get('sample_data', current_sample['data'][sensor])
            
        ego_pose = nusc.get('ego_pose', lidar['ego_pose_token'])
        position = ego_pose['translation']
        orientation = ego_pose['rotation']
        timestamp = ego_pose['timestamp']
    
        # Store the position, orientation, and timestamp
        ego_poses[current_sample['token']]={
            'position': position,
            'orientation': orientation,
            'timestamp': timestamp
        }
    
        # Move on to the next sample
        if current_sample['next'] == '':
            break
        current_sample = nusc.get('sample', current_sample['next'])
    return ego_poses
import pandas as pd

def getObjectPoses1(nusc, ego_poses,scene, sensor='LIDAR_TOP'):
    timestamps_list = []
    poses_list = []
    labels = {}
    
    first_sample_token = scene['first_sample_token']
    current_sample = nusc.get('sample', first_sample_token)

    for i, time in ego_poses.items():
        timestamps_list.append(['ego',time["timestamp"]])
    for i, state in ego_poses.items():
        poses_list.append(['ego',state["position"]])
    while current_sample is not None:
        lidar = nusc.get('sample_data', current_sample['data'][sensor])
        
        for annotation_token in current_sample['anns']:
            annotation = nusc.get('sample_annotation', annotation_token)
            object_token = annotation['instance_token']
            
            # Store the timestamp, position, and label
            timestamps_list.append([object_token, current_sample['timestamp']])
            poses_list.append([object_token, annotation['translation']])
            labels[object_token] = annotation['category_name']

        # Move on to the next sample
        if current_sample['next'] == '':
            break
        current_sample = nusc.get('sample', current_sample['next'])

    # Convert lists to DataFrames and group by object_token
    timestamps_df = pd.DataFrame(timestamps_list).groupby([0])
    poses_df = pd.DataFrame(poses_list).groupby([0])

    # Convert grouped DataFrames to dictionaries
    timestamps_dict = {group_name: group[[1]].values.tolist() for group_name, group in timestamps_df}
    object_poses = {group_name: group[[1]].values.tolist() for group_name, group in poses_df}

    return object_poses, timestamps_dict, labels

def getObjectPoses(nusc,scene,sensor = 'LIDAR_TOP'):
    object_poses = {}
    
    
    
    first_sample_token = scene['first_sample_token']
    current_sample = nusc.get('sample', first_sample_token)
    timestamps = []
    poses=[]    
    labels={}
    while current_sample is not None:

        # Get ego_pose for the current sample
        lidar = nusc.get('sample_data', current_sample['data'][sensor])
        for annotation_token in current_sample['anns']:
            annotation = nusc.get('sample_annotation', annotation_token)
            object_token = annotation['instance_token']
            position = annotation['translation']
            orientation = annotation['rotation']
            timestamps.append([object_token,current_sample['timestamp'],current_sample['token']])
            
             
             # Store the position, orientation, and timestamp
            poses.append([object_token, position,current_sample['token']])
            labels[object_token]=  annotation['category_name']
        # Move on to the next sample
        if current_sample['next'] == '':
            break
        current_sample = nusc.get('sample', current_sample['next'])
    timestamps=pd.DataFrame(timestamps).groupby([0])
    poses=pd.DataFrame(poses).groupby([0])

    timestamps = {group_name: group[[1,2]] for group_name, group in timestamps}
    poses = {group_name: group[[1,2]] for group_name, group in poses}

    return poses,timestamps,labels
        
    
def get_low_level_action_mouvement(velocity, acceleration, angular_velocity):
    '''
    "Stop": When velocity is close to zero and acceleration is minimal.
    "Forward": When velocity is positive and angular velocity is close to zero.
    "Reverse": When velocity is negative and angular velocity is close to zero.
    "Turn Right": When angular velocity is positive and velocity is positive or close to zero.
    "Turn Left": When angular velocity is negative and velocity is positive or close to zero.
    "Accelerate": When acceleration is positive and velocity is positive or close to zero.
    "Decelerate": When acceleration is negative and velocity is positive or close to zero.
    '''
    velocity_magnitude = np.linalg.norm(velocity)
    acceleration_magnitude = np.linalg.norm(acceleration)
    angular_velocity_magnitude = np.linalg.norm(angular_velocity)

    if velocity_magnitude < 0.4 and acceleration_magnitude < 0.4:
        return "Stop"
    elif acceleration_magnitude > 0 and (velocity_magnitude > 0 or abs(velocity_magnitude) < 0.4):
        return "Accelerate"
    elif acceleration_magnitude < 0 and (velocity_magnitude > 0 or abs(velocity_magnitude) < 0.4):
        return "Decelerate"
    elif velocity_magnitude > 0 and angular_velocity_magnitude < 0.4:
        return "Forward"
    elif velocity_magnitude < 0 and angular_velocity_magnitude < 0.4:
        return "Reverse"
    else:
        return "cruising"
def get_low_level_action_steering(velocity, acceleration, angular_velocity):
    '''
    "Stop": When velocity is close to zero and acceleration is minimal.
    "Forward": When velocity is positive and angular velocity is close to zero.
    "Reverse": When velocity is negative and angular velocity is close to zero.
    "Turn Right": When angular velocity is positive and velocity is positive or close to zero.
    "Turn Left": When angular velocity is negative and velocity is positive or close to zero.
    "Accelerate": When acceleration is positive and velocity is positive or close to zero.
    "Decelerate": When acceleration is negative and velocity is positive or close to zero.
    '''
    velocity_magnitude = np.linalg.norm(velocity)
    acceleration_magnitude = np.linalg.norm(acceleration)
    angular_velocity_magnitude = np.linalg.norm(angular_velocity)


    if angular_velocity_magnitude > 0 and (velocity_magnitude > 0 or abs(velocity_magnitude) < 0.2):
        return "Turn Right"
    elif angular_velocity_magnitude < 0 and (velocity_magnitude > 0 or abs(velocity_magnitude) < 0.2):
        return "Turn Left"

    else:
        return "Unknown"
def get_actions(ego_poses):
    velocities = compute_velocity([*ego_poses.values()])
    accelerations = compute_acceleration(velocities, [*ego_poses.values()])
    angular_velocities = compute_angular_velocity([*ego_poses.values()])
    low_level_actions={}
    cp=1
    for  velocity, acceleration, angular_velocity in zip(velocities,accelerations,angular_velocities):
        low_level_actions[cp]=get_low_level_action_mouvement(velocity, acceleration, angular_velocity) 
        cp+=1
    return low_level_actions