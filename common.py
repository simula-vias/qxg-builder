import itertools
import json
import time
import operations
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw
import math
from pyquaternion import Quaternion
from numpy.linalg import inv

from Action_Extraction import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def testing_network(results, prefix1, prefix2, scene):
    for frame in results[prefix1 + str(scene)].keys():
        for key in results[prefix1 + str(scene)][frame][0].keys():
            l1 = set(results[prefix1 + str(scene)][frame][0][key])
            l2 = set(results[prefix2 + str(scene)][frame][0][key])

            if l1 != l2:
                return False

    return True

def getCoord(coords):
    #coords = (*rect1.get_bbox().min, *rect1.get_bbox().max)
    centerx, centery = (np.average(coords[:2]), np.average(coords[2:]))

    return centerx, centery


def SameQuarter(o1, o2):
    # +-----------------------------------+
# |              |                    |
# |   First      |      Second        |
# |   Quarter   case3      Quarter       |
# |   (+, +)     |      (-, +)        |
# |              |                    |
# +-----case2---------------case2---------+
# |              |                    |
# |   Fourth     |      Third         |
# |   Quarter    case3      Quarter       |
# |   (+, -)     |      (-, -)        |
# |              |                    |
# +-----------------------------------+
    xc, yc = 0, 0
    (xi, yi) = getCoord(o1)
    (xj, yj) = getCoord(o2)
    if (xi > xc) and (xj > xc) and (yi > yc) and (yj > yc):
        return 1  # first quarter;
    if (xi < xc) and (xj < xc) and (yi > yc) and (yj > yc):
        return 2  # second quarter;
    if (xi < xc) and (xj < xc) and (yi < yc) and (yj < yc):
        return 3  # third quarter;
    if (xi > xc) and (xj > xc) and (yi < yc) and (yj < yc):
        return 4  # fourth quarter;
    if ((yi == 0) and abs(xi - xj) != 0) or ((yj == 0) and abs(xi - xj) != 0):
        return 5  # case2 in figure 4;
    if ((xi == 0) and abs(yi - yj) != 0) or ((xj == 0) and abs(yi - yj) != 0):
        return 6  # case3 in figure 4;
    return [getQuarter(o1),getQuarter(o2)]

def getQuarter(o1):
    xc, yc = 0, 0
    (xi, yi) = getCoord(o1)
    if (xi > xc)  and (yi > yc) :
        return 1  # first quarter;
    if (xi < xc)  and (yi > yc) :
        return 2  # second quarter;
    if (xi < xc) and (yi < yc) :
        return 3  # third quarter;
    if (xi > xc)  and (yi < yc) :
        return 4  # fourth quarter;
    if ((yi == 0) ):
        return 5  # case2 in figure 4;
    if ((xi == 0) ):
        return 6  # case3 in figure 4;
    return 0
def get_spatial_relations(boxes, o_i, o_j, rels):
    learned = set()
    bb1 = list(boxes[o_i])
    bb2 = list(boxes[o_j])

    for r in rels:
        if isinstance(r, tuple):
            for r1 in list(r):
                answer = operations.compute_RA_Algebra(bb1, bb2, r1)
                if answer:
                    learned.add(r1)
        else:
            answer = operations.compute_RA_Algebra(bb1, bb2, r1)
            if answer:
                learned.add(r)
    return learned


def get_direction(o_i,o_j,ego_poses,s,t):
    
    object1_poses = []
    object2_poses = []
    timestamps = []

    timestamps1 = []
    if o_i =='ego':
        
        for i, time in ego_poses.items():
            timestamps.append(time['timestamp'])
        for i, state in ego_poses.items():

             object1_poses.append(state['position']) 
    else:
        for i, time in t[o_i].iterrows():
            timestamps.append(time[1])
        for i,state in s[o_i].iterrows():

             object1_poses.append(state[1])     
            
    if o_j =='ego':
         for i, time in ego_poses.items():
             timestamps1.append(time['timestamp'])
         for i, state in ego_poses.items():
        
              object2_poses.append(state['position']) 
        
    else :
        for i, time in t[o_j].iterrows():
            timestamps1.append(time[1])
        
        # Replace `object_poses` with the appropriate object poses
        for i,state in s[o_j].iterrows():
    
            object2_poses.append(state[1])  # Replace `object_poses` with the appropriate object poses

    # Compute velocities of both objects
    object1_velocity = compute_velocity_object(object1_poses, timestamps)
    object2_velocity = compute_velocity_object(object2_poses, timestamps1)
    min_len = min(len(object1_poses), len(object2_poses), len(object1_velocity), len(object2_velocity))

    # Cut the object poses and velocities to match the number of elements of the object with fewer elements
    object1_poses = object1_poses[:min_len]
    object2_poses = object2_poses[:min_len]
    object1_velocity = object1_velocity[:min_len]
    object2_velocity = object2_velocity[:min_len]
    
    # Calculate the relative vector of positioning between the two objects
    relative_position = np.array(object2_poses) - np.array(object1_poses)
    # Calculate the relative vector of velocity between the two objects

    relative_velocity = np.array(object2_velocity) - np.array(object1_velocity)

    # Check the direction of the relative vector
    direction = ""
    
    # Compute the dot product between the relative velocity and position vectors
    dot_product = np.sum(relative_velocity * relative_position, axis=1)
    
   # Determine the direction based on the dot product
    same_direction_indices = np.where(dot_product > 0)[0]
    opposite_direction_indices = np.where(dot_product < 0)[0]

    directions = np.full(len(dot_product), "perpendicular or stationary")
    directions[same_direction_indices] = "same direction"
    directions[opposite_direction_indices] = "opposite direction"


    return list(directions)

def get_distance(o1,o2):
    centroid_1 = o1['translation'][:2]
    centroid_2 = o2['translation'][:2]
    
    # Calculate the Euclidean distance between the centroids
    distance = math.sqrt((centroid_1[0] - centroid_2[0]) ** 2 + (centroid_1[1] - centroid_2[1]) ** 2)
    return distance

def get_mov(ego_poses,s,t):
    

    ego_poses_=[]
    timestamps=[]
    object_poses=[]

    for i,time in t.iterrows():
        ego_poses_.append(ego_poses[time[2]])
        timestamps.append(time[1])
    for i,state in s.iterrows():
        object_poses.append(state[1])
    object_velocity=compute_velocity_object(object_poses,timestamps)
    ego_velocity=compute_velocity(ego_poses_)
    length=len(ego_velocity)
    attributes=[]
    
    for i in range(0,length,3):
        is_dynamic=False
        # Get the velocity components of the object in the current frame
        x1=np.array(object_velocity)[i:i+3]
        x2=np.array(ego_velocity)[i:i+3]

        relative_velocity = np.array(x1) - np.array(x2)
        # Set the threshold for dynamic/static classification
        threshold = 0.1
        speed = np.linalg.norm(x1)
        # Check if any of the velocity components exceed the threshold
        if speed > threshold:
            is_dynamic = True
            
        attributes.append([i,is_dynamic])

    
    return attributes
def get_relation(bb1, bb2):
    return (
        get_allen((bb1[0], bb1[2]), (bb2[0], bb2[2])) + "x",
        get_allen((bb1[1], bb1[3]), (bb2[1], bb2[3])) + "y",
    )


def get_allen(i1, i2):
    if i2[1] < i1[0]:
        return "BI"
    if i2[1] == i1[0]:
        return "MI"
    if i2[0] < i1[0] < i2[1] and i1[0] < i2[1] < i1[1]:
        return "OI"
    if i2[0] == i1[0] and i2[1] < i1[1]:
        return "SI"
    if i1[0] < i2[0] < i1[1] and i1[0] < i2[1] < i1[1]:
        return "DI"
    if i1[0] < i2[0] < i1[1] and i2[1] == i1[1]:
        return "FI"
    if i2[0] == i1[0] and i2[1] == i1[1]:
        return "E"
    if i1[1] < i2[0]:
        return "B"
    if i1[1] == i2[0]:
        return "M"
    if i1[0] < i2[0] < i1[1] and i2[0] < i1[1] < i2[1]:
        return "O"
    if i1[0] == i2[0] and i1[1] < i2[1]:
        return "S"
    if i2[0] < i1[0] < i2[1] and i2[0] < i1[1] < i2[1]:
        return "D"
    if i2[0] < i1[0] < i2[1] and i1[1] == i2[1]:
        return "F"


def QXGBUILDER(boxes,metadata, frame_idx=0, initial_graph={}):
    begin = time.time()
    learned = initial_graph

    for o_i, o_j in itertools.combinations(boxes, 2):
        rels = learned.get((o_i, o_j), [])
        pair = (metadata[o_i],metadata[o_j])
        rels.append((frame_idx, get_relation(boxes[o_i], boxes[o_j]),
                     get_direction(o_i,o_j,metadata['ego_poses'],metadata['states'],metadata['timestamps']),
                     get_distance(metadata[o_i], metadata[o_j]),SameQuarter(boxes[o_i], boxes[o_j])))
        learned[(o_i, o_j)] = rels
        
    end = time.time()
    return [learned, (end - begin)]


def BruteForce(boxes, frame_idx, initial_graph={}):
    all_rels = "Bx|BIx|Dx|DIx|Ex|Fx|FIx|Mx|MIx|Ox|OIx|Sx|SIx|By|BIy|Dy|DIy|Ey|Fy|FIy|My|MIy|Oy|OIy|Sy|SIy".split(
        "|"
    )
    all_possibilities = []
    for c in itertools.combinations(all_rels, 2):
        if "x" in c[0] and "y" in c[1]:
            all_possibilities.append(c)

    learned = initial_graph
    begin = time.time()
    for o_i, o_j in itertools.combinations(boxes, 2):
        rels = learned.get((o_i, o_j), [])
        rels.append(
            (frame_idx, get_spatial_relations(boxes, o_i, o_j, all_possibilities))
        )
        learned[(o_i, o_j)] = rels

    end = time.time()

    return [learned, (end - begin)]


def get_spatial_relations(boxes, o_i, o_j, rels):
    learned = set()
    bb1 = list(boxes[o_i])
    bb2 = list(boxes[o_j])

    for r in rels:
        if isinstance(r, tuple):
            for r1 in list(r):
                answer = operations.compute_RA_Algebra(bb1, bb2, r1)
                if answer:
                    learned.add(r1)
        else:
            answer = operations.compute_RA_Algebra(bb1, bb2, r1)
            if answer:
                learned.add(r)
        # if len(learned)==2:
        # break
    # if len(learned)>2:
    # learned=get_relation(bb1, bb2)
    return tuple(sorted(learned, key=lambda s: s[-1]))



def Build_Rectangle(bbox):
    left1, bottom1, width1, height1 = bbox[0], bbox[1], bbox[2], bbox[3]
    return (left1, bottom1, left1 + width1, bottom1 + height1)


def num_nodes(edges):
    n = []
    for e in edges.keys():
        n.extend(e)
    return len(set(n))
