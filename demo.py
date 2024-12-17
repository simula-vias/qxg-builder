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


'''
mode = 'temporal'
version = "v1.0-mini"
dataroot = "/home/nassim/Desktop/Self-DrivingCara-XAI/data/sets/nuscenes"
sensor = 'LIDAR_TOP'



nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

scenes = nusc.scene
results = {}
'''
def components(o_i,ego_poses,s,t):
    object1_poses = []
    timestamps = []

    if o_i == "ego":
        for i, time in ego_poses.items():
            timestamps.append(time["timestamp"])
        for i, state in ego_poses.items():
            object1_poses.append(state["position"])
    else:
        for i, time in t[o_i].iterrows():
            timestamps.append(time[1])
        for i, state in s[o_i].iterrows():
            object1_poses.append(state[1])

def launch(scenes,nusc,sensor,mode,results):
    for nb, s in enumerate(scenes):
        #if nb==0:
            print(f"scene {nb+1}/{len(scenes)}")
            name = s["name"]
        
            frames = getBBoxFromSensor(sensor, s, nusc)
            #print_boxes(frames)
    
            res_bf = {}
            res_sf = {}
            
            ego_poses= getEgoPoses(nusc, s,sensor)
            states,timestamps,labels= getObjectPoses1(nusc,ego_poses, s,sensor)

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
                # This can be optimized by get the position of objects at each frame only
                #object_poses,timestamps_poses={},{}
                #for o_i in boxes:
                    #try:
                        #object_poses[o_i]
                        #continue
    
                    #except:
                        #object_poses[o_i],timestamps_poses[o_i]=components(o_i,metadata["ego_poses"],states,timestamps)
    
    
                metadata['object_poses']=states
                metadata['timestamps_poses']=timestamps
    
                grow_graph = mode == "temporal" and cp > 0
                
                res_sf[cp] = QXGBUILDER(
                    boxes,metadata,1, cp, dict(res_sf[cp - 1][0]) if grow_graph else {}
                )
                
                nodes = num_nodes(res_sf[cp][0])
                edges = len(res_sf[cp][0])
        
            #for o in states:
                #dynamicity[o+'-'+labels[o]]=get_mov(ego_poses,states[o],timestamps[o])
            results[f"SmartForce_{name}"] = [res_sf,get_actions(ego_poses)]
            print(name)
            #break
    
    print("Done")












