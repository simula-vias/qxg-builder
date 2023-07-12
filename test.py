#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 10:18:49 2023

@author: nassim
"""


import pickle

from nuscenes.nuscenes import NuScenes
from common import *
from Nuscenes_parser import *
from Action_Extraction import *
import argparse
import os



mode = 'temporal'
version = "v1.0-mini"
dataroot = "/home/nassim/Desktop/Self-DrivingCara-XAI/data/sets/nuscenes"
sensor = 'LIDAR_TOP'

nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

scenes = nusc.scene

for nb, s in enumerate(scenes):
    if nb==4:
        print(f"scene {nb+1}/{len(scenes)}")
        name = s["name"]
    
        frames = getBBoxFromSensor(sensor, s, nusc)
        res_bf = {}
        res_sf = {}
        states,timestamps,labels= getObjectPoses(nusc, s,sensor)
        
        ego_poses= getEgoPoses(nusc, s,sensor)

        dynamicity={}
        for cp, f in enumerate(frames):
            boxes = {}
            metadata={}
            for o in f:
                boxes[o["instance_token"]] = Build_Rectangle(o["bbox"][0])
                metadata[o["instance_token"]] = o
            current_sample = nusc.get('sample', o['sample_token'])
            lidar = nusc.get('sample_data', current_sample['data'][sensor])

            metadata['ego'] =  nusc.get('ego_pose', lidar['ego_pose_token'])
            metadata['states']=states
            metadata['timestamps']=timestamps
            metadata['ego_poses']=ego_poses

            
            if len(boxes) > 0:
                boxes["ego"] = Build_Rectangle(o["ego_box"])
            else:
                print(f"scene {nb}: no objects in frame {cp} for sensor {sensor}")
    
            grow_graph = mode == "temporal" and cp > 0
    
            res_sf[cp] = QXGBUILDER(
                boxes,metadata,1, cp, dict(res_sf[cp - 1][0]) if grow_graph else {}
            )
            
            nodes = num_nodes(res_sf[cp][0])
            edges = len(res_sf[cp][0])
    
        for o in states:
            dynamicity[o+'-'+labels[o]]=get_mov(ego_poses,states[o],timestamps[o])
        results = {}
        results[f"SmartForce_{name}"] = res_sf
        break

print("Done")
ego_poses=[*ego_poses.values()]
velocities = compute_velocity(ego_poses)
accelerations = compute_acceleration(velocities, ego_poses)
angular_velocities = compute_angular_velocity(ego_poses)
low_level_actions=[]
#for  velocity, acceleration, angular_velocity in zip(velocities,accelerations,angular_velocities):
#low_level_actions.append(get_low_level_action_mouvement(velocity, acceleration, angular_velocity)) 

