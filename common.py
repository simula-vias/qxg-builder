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
from data_builder import *
import math

def testing_network(results, prefix1, prefix2, scene):
    for frame in results[prefix1 + str(scene)].keys():
        for key in results[prefix1 + str(scene)][frame][0].keys():
            l1 = set(results[prefix1 + str(scene)][frame][0][key])
            l2 = set(results[prefix2 + str(scene)][frame][0][key])

            if l1 != l2:
                return False

    return True

def Translate(relation):
            Translation = {
                   'Dx|Sx|Fx|Ex|Dy|Sy|Fy|Ey':'B' ,
                   'Dx|Sx|Fx|Ex|My|By':'S',
                   'Dx|Sx|Fx|Ex|MIy|BIy':'N',
                   'MIx|BIx|Dy|Sy|Fy|Ey':'E',
                   'Mx|Bx|Dy|Sy|Fy|Ey':'W',
                   'MIx|BIx|MIy|BIy': 'NE',
                   'Mx|Bx|MIy|BIy': 'NW',
                   'MIx|BIx|My|By':'SE',
                   'Mx|Bx|My|By':'SW',
                
                
                
                   'FIx|Ox|My|By':'S:SW' ,
                   'SIx|OIx|My|By':'S:SE',
                   'FIx|Ox|MIy|BIy':'N:NW',
                   'SIx|OIx|MIy|BIy':'N:NE',
                   'FIx|Ox|Dy|Sy|Fy|Ey':'B:W',
                   'SIx|OIx|Dy|Sy|Fy|Ey': 'B:E',
                   'Dx|Sx|Fx|Ex|FIy|Oy':'B:S',
                   'Dx|Sx|Fx|Ex|SIy|OIy':'B:N',
                   'Mx|Bx|FIy|Oy':'W:SW',
                   'Mx|Bx|SIy|OIy':'W:NW',
                   'MIx|BIx|FIy|Oy':'E:SE',
                   'MIx|BIx|SIy|OIy':'E:NE',
                   'DIx|My|By':'S:SW:SE',
                   'DIx|MIy|BIy':'N:NW:NE',
                   'DIx|Dy|Sy|Fy|Ey':'B:W:E',
                   'Dx|Sx|Fx|Ex|DIy':'B:N:S',
                  'Mx|Bx|DIy':'W:NW:SW',
                   'MIx|BIx|DIy':'E:NE:SE',
                   'Ox|FIx|Oy|FIy':'B:S:SW:W',
                   'Ox|FIx|SIy|OIy':'B:W:NW:N',
                   'SIx|OIx|Oy|FIy':'B:S:E:SE',
                   'SIx|OIx|SIy|OIy':'B:N:NE:E',
                  'Ox|FIx|DIy':'B:S:SW:W:NW:N',
                  'SIx|OIx|DIy':'B:S:SE:E:NE:N',
                   'DIx|FIy|Oy':'B:S:SW:W:E:SE',
                   'DIx|SIy|OIy':'B:W:NW:N:NE:E',
                  'DIx|DIy':'B:S:SW:W:NW:N:NE:E:SE'
                  

                           }
            for key in Translation.keys():
                 if relation[0] in key.split('|') and relation[1] in key.split('|') :
                     return Translation[key]
            return relation
                
def getCoord(coords):
    # coords = (*rect1.get_bbox().min, *rect1.get_bbox().max)
    centerx, centery = (np.average(coords[:2]), np.average(coords[2:]))

    return centerx, centery


def SameQuarter(o1, o2):
    
    #todo: introduce slack
    # +-----------------------------------+
    # |              |                    |
    # |   Second     |      First         |
    # |   Quarter   case3      Quarter    |
    # |   (-, +)     |      (+, +)        |
    # |              |                    |
    # +-----case2----0----------case2-----+
    # |              |                    |
    # |   Third      |      Fourth        |
    # |   Quarter   case3      Quarter    |
    # |   (-, -)     |      (+, -)        |
    # |              |                    |
    # +-----------------------------------+
    xc, yc = 0, 0
    (xi, yi) = getCoord(o1)
    (xj, yj) = getCoord(o2)

    return [getQuarter(o1), getQuarter(o2)]


def getQuarter(o1, slack=1):
    xc, yc = 0, 0
    (xi, yi) = getCoord(o1)
    
    # Check if the object is within the slack range of the axes
    if abs(yi) <= slack:
        return 5  # case2 in figure 4;
    if abs(xi) <= slack:
        return 6  # case3 in figure 4;

    if (xi > xc) and (yi > yc):
        return 1  # first quarter;
    if (xi < xc) and (yi > yc):
        return 2  # second quarter;
    if (xi < xc) and (yi < yc):
        return 3  # third quarter;
    if (xi > xc) and (yi < yc):
        return 4  # fourth quarter;

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

def compute_qtcb_relation(prev_distance, current_distance):
    """
    Compute the QTCB relation based on the change in distance.
    """
    if prev_distance < current_distance:
        return "Moving Away"  # Moving away
    elif prev_distance > current_distance:
        return "Moving Towards"  # Moving towards
    else:
        return "Startionary"  # Stable

def qtcb_relations(positions_k, positions_l):
    """
    Compute QTCB relations for two objects based on their positions.
    
    Args:
    - positions_k: List of positions of object k.
    - positions_l: List of positions of object l.
    
    Returns:
    - List of QTCB relations for each object relative to the other.
    """
    min_len = min(
        len(positions_k),
        len(positions_l)
    )

    # Cut the object poses and velocities to match the number of elements of the object with fewer elements
    positions_k = positions_k[:min_len]
    positions_l = positions_l[:min_len]


    if len(positions_k) != len(positions_l):
        raise ValueError("Both objects must have the same number of positions.")
    
    relations_k = []
    relations_l = []
    for i in range(1, len(positions_k)):
        centroid_1_prev = positions_k[i-1]
        centroid_2_prev = positions_l[i-1]
        centroid_1 = positions_k[i]
        centroid_2 = positions_l[i]
        # Calculate the Euclidean distance between the centroids
        prev_distance_k = np.linalg.norm(np.array(centroid_1_prev) - np.array(centroid_2_prev))

        current_distance_k = np.linalg.norm(np.array(centroid_1) - np.array(centroid_2))
        
        prev_distance_l = np.linalg.norm(np.array(centroid_2_prev) - np.array(centroid_1_prev))
        current_distance_l =  np.linalg.norm(np.array(centroid_2) - np.array(centroid_1))
        
        relation_k = compute_qtcb_relation(prev_distance_k, current_distance_k)
        relation_l = compute_qtcb_relation(prev_distance_l, current_distance_l)
        
        relations_k.append(relation_k)
        relations_l.append(relation_l)
    
    return relations_k, relations_l




def get_direction(object1_poses,object2_poses,timestamps,timestamps1):
    
    object1_velocity = compute_velocity_object(object1_poses, timestamps)
    object2_velocity = compute_velocity_object(object2_poses, timestamps1)
    min_len = min(
        len(object1_poses),
        len(object2_poses),
        len(object1_velocity),
        len(object2_velocity),
    )

    # Cut the object poses and velocities to match the number of elements of the object with fewer elements
    object1_poses1 = object1_poses[:min_len]
    object2_poses1 = object2_poses[:min_len]
    object1_velocity1 = object1_velocity[:min_len]
    object2_velocity1 = object2_velocity[:min_len]


    # Calculate the relative vector of positioning between the two objects
    relative_position = np.array(object2_poses1) - np.array(object1_poses1)
    # Calculate the relative vector of velocity between the two objects

    relative_velocity = np.array(object2_velocity1) - np.array(object1_velocity1)
    if len(relative_position) <=0 or len(relative_velocity)<=0 :

       return None
    # Check the direction of the relative vector
    direction = ""
    #print(np.array(object2_poses).shape)
    #print(np.array(object1_poses).shape)
    #print(np.array(object2_velocity).shape)
    #print(np.array(object1_velocity).shape)
    # Compute the dot product between the relative velocity and position vectors
    dot_product = np.sum(relative_velocity * relative_position, axis=1)

    # Determine the direction based on the dot product
    same_direction_indices = np.where(dot_product > 0)[0]
    opposite_direction_indices = np.where(dot_product < 0)[0]

    directions = np.full(len(dot_product), "stationary")
    directions[same_direction_indices] = "same direction"
    directions[opposite_direction_indices] = "opposite direction"

    return directions

def get_distance(o1, o2):
    centroid_1 = o1["translation"][:2]
    centroid_2 = o2["translation"][:2]

    # Calculate the Euclidean distance between the centroids
    distance = math.sqrt(
        (centroid_1[0] - centroid_2[0]) ** 2 + (centroid_1[1] - centroid_2[1]) ** 2
    )
    if distance<5:
        return 'very close'
    if distance>=5 and distance<=30:
        return 'close'
    if distance>30 and distance<=50:
        return 'far'
    else:
        return 'very far'


def get_mov(ego_poses, s, t):
    ego_poses_ = []
    timestamps = []
    object_poses = []

    for i, time in t.iterrows():
        ego_poses_.append(ego_poses[time[2]])
        timestamps.append(time[1])
    for i, state in s.iterrows():
        object_poses.append(state[1])
    object_velocity = compute_velocity_object(object_poses, timestamps)
    ego_velocity = compute_velocity(ego_poses_)
    length = len(ego_velocity)
    attributes = []

    for i in range(0, length, 3):
        is_dynamic = False
        # Get the velocity components of the object in the current frame
        x1 = np.array(object_velocity)[i : i + 3]
        x2 = np.array(ego_velocity)[i : i + 3]

        relative_velocity = np.array(x1) - np.array(x2)
        # Set the threshold for dynamic/static classification
        threshold = 0.1
        speed = np.linalg.norm(x1)
        # Check if any of the velocity components exceed the threshold
        if speed > threshold:
            is_dynamic = True

        attributes.append([i, is_dynamic])

    return attributes


def get_relation(bb1, bb2, slack):
    return (
        get_allen((bb1[0], bb1[2]), (bb2[0], bb2[2]), slack) + "x",
        get_allen((bb1[1], bb1[3]), (bb2[1], bb2[3]), slack) + "y",
    )


def get_allen(i1, i2, slack):
    assert i1[0] < i1[1]
    assert i2[0] < i2[1]

    # i1 _ rel _ i2
    start_before_start = i1[0] < i2[0] - slack
    start_before_end = i1[0] < i2[1] - slack
    start_meets_start = i2[0] - slack <= i1[0] <= i2[0] + slack
    start_meets_end = i2[1] - slack <= i1[0] <= i2[1] + slack
    start_after_start = i1[0] > i2[0] + slack
    start_after_end = i1[0] > i2[1] + slack

    end_before_start = i1[1] < i2[0] - slack
    end_before_end = i1[1] < i2[1] - slack
    end_meets_start = i2[0] - slack <= i1[1] <= i2[0] + slack
    end_meets_end = i2[1] - slack <= i1[1] <= i2[1] + slack
    end_after_start = i1[1] > i2[0] + slack
    end_after_end = i1[1] > i2[1] + slack

    if start_meets_start and end_meets_end:
        return "E"

    if start_meets_start and end_before_end:
        return "S"

    if start_meets_start and end_after_end:
        return "SI"

    if start_before_start and end_after_start and end_before_end:
        return "O"

    if start_after_start and start_before_end and end_after_end:
        return "OI"

    if start_after_start and end_meets_end:
        return "F"

    if start_before_start and end_meets_end:
        return "FI"

    if end_meets_start:
        return "M"

    if start_meets_end:
        return "MI"

    if end_before_start:
        return "B"

    if start_after_end:
        return "BI"

    if start_after_start and end_before_end:
        return "D"

    if start_before_start and end_after_end:
        return "DI"

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

   
    return object1_poses,timestamps,
def QXGBUILDER(boxes, metadata, slack, frame_idx=0, initial_graph={}):
    begin = time.time()
    learned = initial_graph
    binary_rep=[]

    for o_i, o_j in itertools.combinations(boxes, 2):
        rels = learned.get((o_i, o_j), [])
        pair = (metadata[o_i], metadata[o_j])
        RA=get_relation(boxes[o_i], boxes[o_j], slack)
        RA_binary=encode_RA(RA)
        
        if len(metadata['object_poses'][o_i])>2 and len(metadata['object_poses'][o_j])>2:
            relations_k, relations_l =qtcb_relations(metadata['object_poses'][o_i],metadata['object_poses'][o_j])
            try:
                DIR=relations_k[frame_idx]
            except:
                DIR=relations_k[len(relations_k)-1]
        else:
            relations_k='Stationary'
            DIR=relations_k
        #DIR1=get_direction(metadata['object_poses'][o_i],metadata['object_poses'][o_j],metadata['timestamps_poses'][o_i],metadata['timestamps_poses'][o_j])[frame_idx]
        DIR_binary=encode_QTC(DIR)
        #DIR1_binary=encode_DIR(DIR1)

        DIS=get_distance(metadata[o_i], metadata[o_j])
        STAR=SameQuarter(boxes[o_i], boxes[o_j])
        STAR_binary=encode_STAR(STAR)
        
        rels.append(
            (
                frame_idx,
                RA,
                DIR,
                #DIR1,
                DIS,
                STAR,
                RA_binary,
                DIR_binary,
                #DIR1_binary,
                STAR_binary
                
            )
        )
        if 'ego' in o_i:
            row=[(o_i, metadata[o_j]['category_name']+'_'+o_j),frame_idx,DIS,Translate(RA)]

            learned[(o_i, metadata[o_j]['category_name']+'_'+o_j)] = rels
        if 'ego' in o_j:
            row=[(metadata[o_i]['category_name']+'_'+o_i, o_j),frame_idx,DIS,Translate(RA)]

            learned[(metadata[o_i]['category_name']+'_'+o_i, o_j)] = rels

        else:
            row=[(metadata[o_i]['category_name']+'_'+o_i, metadata[o_j]['category_name']+'_'+o_j),frame_idx,DIS,Translate(RA)]

            learned[(metadata[o_i]['category_name']+'_'+o_i, metadata[o_j]['category_name']+'_'+o_j)] = rels
        
        
        '''for e in list(RA_binary.values()):
            row.append(e)
        for e1 in list(DIR_binary.keys()):
            if DIR_binary[e1]==1:
                row.append(e1)
        #for e1 in list(DIR1_binary.keys()):
            #if DIR1_binary[e1]==1:
                #row.append(e1)'''
        row.append(DIR)
        for e2 in STAR:
            row.append(e2)
       
        binary_rep.append(row)
    end = time.time()
    
    return [learned,binary_rep, (end - begin)]


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
