#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:26:15 2023

@author: nassim
"""
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from nuscenes.utils.data_classes import Box
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def BBox_from_Cameras(sensor, s, nusc):
    my_scene = s
    collected = []

    first_sample_token = my_scene["first_sample_token"]
    sample = nusc.get("sample", first_sample_token)

    while True:
        metadata = []
        cam_front_data = nusc.get("sample_data", sample["data"][sensor])

        # Get the boxes and camera calibration
        # im_data, boxes, camera_intrinsic = nusc.get_sample_data(cam_front_data['token'])
        cam_calib = nusc.get(
            "calibrated_sensor", cam_front_data["calibrated_sensor_token"]
        )
        fig, ax = plt.subplots()
        ego_pose = nusc.get("ego_pose", cam_front_data["ego_pose_token"])
        ego_size = (4.76, 1.93, 1.72)  # length, width, height

        # Create a box for the ego vehicle
        ego_box_camera = Box(
            cam_calib["translation"], ego_size, Quaternion(cam_calib["rotation"])
        )
        corners_ego = ego_box_camera.bottom_corners()
        emin_x, emax_x = np.min(corners_ego[0, :]), np.max(corners_ego[0, :])
        emin_y, emax_y = np.min(corners_ego[1, :]), np.max(corners_ego[1, :])
        ewidth = emax_x - emin_x
        eheight = emax_y - emin_y
        ax.add_patch(Rectangle((emin_x, emin_y), ewidth, eheight, fill=False, edgecolor='b'))

        for t in sample["anns"]:
            my_annotation_metadata = nusc.get("sample_annotation", t)

            _, boxes, camera_intrinsic = nusc.get_sample_data(
                sample["data"][sensor], selected_anntokens=[t]
            )

            for box in boxes:
                # Check if the box is visible
                if box_in_image(
                    box,
                    camera_intrinsic,
                    (cam_front_data["width"], cam_front_data["height"]),
                    vis_level=BoxVisibility.ANY,
                ):
                    # Rotate the box to the ego vehicle's coordinate frame
                    box.rotate(Quaternion(cam_calib["rotation"]))
                    # Plot bird's eye view
                    corners = box.bottom_corners()
                    min_x, max_x = np.min(corners[0, :]), np.max(corners[0, :])
                    min_y, max_y = np.min(corners[1, :]), np.max(corners[1, :])
                    width = max_x - min_x
                    height = max_y - min_y
                    ax.add_patch(Rectangle((min_x, min_y), width, height, fill=False, edgecolor='red'))
                    # my_annotation_metadata["instance_token"] = box.instance_token
                    my_annotation_metadata["bbox"] = [[min_y, min_x, height, width]]
                    my_annotation_metadata["name"] = box.name
                    my_annotation_metadata["ego_box"] = [
                        emin_y,
                        emin_x,
                        eheight,
                        ewidth,
                    ]

                    metadata.append(my_annotation_metadata.copy())

        ax.set_title('Bird\'s eye view of the 3D bounding boxes')
        ax.set_xlabel('x (meters)')
        ax.set_ylabel('y (meters)')
        ax.axis('equal')
        plt.show()
        fig.savefig('scenes/'+cam_front_data['sample_token']+'_bis.jpg', dpi=300)

        # im_data = np.array(Image.open(im_data))
        nusc.render_sample_data(cam_front_data["token"],out_path='scenes/'
                                +cam_front_data['sample_token']+'.jpg')

        try:
            sample = nusc.get("sample", sample["next"])

        except:
            break
        collected.append(metadata.copy())
    return collected


def rotate_2d_bbox(bbox, angle):
    # Rotation matrix
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Rotate each corner
    return [np.dot(R, corner - bbox.center[:2]) + bbox.center[:2] for corner in bbox.corners()[:2, :].T]

def getLidarBBox_of_Scenes1(s, nusc):
    collected = []

    my_scene = s
    first_sample_token = my_scene["first_sample_token"]
    n = nusc.get("sample", first_sample_token)
    margin = 10

    while True:
        metadata = []

        sensor = "LIDAR_TOP"


        lidar = nusc.get("sample_data", n["data"][sensor])
        
        pose = nusc.get("ego_pose", lidar["ego_pose_token"])
        #nusc.render_sample_data(lidar["token"])

        
        # default car size and position
        w1, l1, h1 =  4.478,1.871, 1.456
        # x, y, z = -12.327768364061617, -0.40654977344377285, 0.664

        # Get the quaternion rotation of the ego vehicle
        ego_quat = Quaternion(pose['rotation'])
        # Compute the yaw angle (rotation around the vertical axis)
        ego_box = [0, 0, w1,l1]
        my_annotation_token = n["anns"]
        
        j = 0
        for t in my_annotation_token:
            bboxes = []
            my_annotation_metadata = nusc.get("sample_annotation", t)
            data_path, boxes, camera_intrinsic = nusc.get_sample_data(
                n["data"][sensor], selected_anntokens=[t]
            )
            w, h, l = my_annotation_metadata["size"]
            dist = np.linalg.norm(
                np.array(pose["translation"])
                - np.array(my_annotation_metadata["translation"])
            )
            idx = "instance_token"
            name = "category_name"
            rotation = "rotation"
            size = "size"
            translation = "translation"
            size = my_annotation_metadata["size"]
            for box in boxes:
                relative_rotation = ego_quat.inverse * box.orientation
                rotated_box = Box(box.center, box.wlh, relative_rotation, name=box.name, token=box.token)

                view: np.ndarray = np.eye(4)
                corners = view_points(rotated_box.corners(), view, False)[:2, :]
                minx, maxx = np.min(corners[0, :]), np.max(corners[0, :])
                miny, maxy = np.min(corners[1, :]), np.max(corners[1, :]) 
                bboxes.append([miny, minx, h, w])
                direction = rotated_box.center


            my_annotation_metadata["bbox"] = bboxes
            my_annotation_metadata["dist"] = dist
            my_annotation_metadata["ego_box"] = ego_box
            my_annotation_metadata["direction"] = direction

            metadata.append(my_annotation_metadata.copy())
            j += 1
        try:
            n = nusc.get("sample", n["next"])

        except:
            break
        collected.append(metadata.copy())
    return collected

def getLidarBBox_of_Scenes(s, nusc):
    collected = []
    my_scene = s
    first_sample_token = my_scene["first_sample_token"]
    n = nusc.get("sample", first_sample_token)
    margin = 10

    while True:
        metadata = []
        sensor = "LIDAR_TOP"
        lidar = nusc.get("sample_data", n["data"][sensor])
        pose = nusc.get("ego_pose", lidar["ego_pose_token"])
        #nusc.render_sample_data(lidar["token"])

        # default car size and position
        w1, l1, h1 =  4.478,1.871, 1.456

        # Create a box representing the ego vehicle in the ego vehicle's coordinate frame
        # The center of the box is at (0, 0, 0), and the orientation is the identity quaternion
        ego_box = [0, 0, w1,l1]
        
        # Compute ego car's rotation matrix
        ref_sd_token = n["data"][sensor]
        
        _, boxes, _ = nusc.get_sample_data(ref_sd_token,  use_flat_vehicle_coordinates=True)
        for t in boxes:
            bboxes = []
            my_annotation_metadata = nusc.get("sample_annotation", t.token)
            w, l, h = t.wlh

            dist = np.linalg.norm(
                np.array(pose["translation"])
                - np.array(my_annotation_metadata["translation"])
            )
           # _, ax = plt.subplots(1, 1, figsize=(9, 9))

            #t.render(ax, view=np.eye(4))
            translation = np.array(pose['translation'])

            view: np.ndarray = np.eye(4)

            corners = view_points(t.corners(), view, False)[:2, :]
             
            bbox_2d = [
                        np.min(corners[0, :]),  # min_x
                        np.min(corners[1, :]),  # min_y
                        l,  # max_x
                        w  # max_y
                    ]
            bboxes.append(bbox_2d)
            
            direction = t.center

            my_annotation_metadata["bbox"] = bboxes
            my_annotation_metadata["corners"] = corners

            my_annotation_metadata["dist"] = dist
            my_annotation_metadata["ego_box"] = ego_box
            my_annotation_metadata["direction"] = direction

            metadata.append(my_annotation_metadata.copy())
            #plt.show()

        try:
            n = nusc.get("sample", n["next"])
        except:
            break
        collected.append(metadata.copy())
    return collected

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patches, text, patheffects
from pyquaternion import Quaternion
import matplotlib.patches as mpatches

def print_boxes(collected,colors = ('b', 'r', 'k'),linewidth: float = 1):
    rels = 'Bx|BIx|Dx|DIx|Ex|Fx|FIx|Mx|MIx|Ox|OIx|Sx|SIx|By|BIy|Dy|DIy|Ey|Fy|FIy|My|MIy|Oy|OIy|Sy|SIy'.split(
        '|')
    cp = 0
    dfs = []
    for c in collected:
        fig1, axis = plt.subplots()
        #ax1.set_xlim([-60, 50])
        #ax1.set_ylim([-60, 50])
        cp += 1
        df = pd.DataFrame.from_dict(c)
        dfs.append(df)
        concat = pd.concat(dfs)
        groups = concat.groupby(['instance_token'])
        result = []
        egobox = np.array(c[0]['ego_box'])


        for d in c:
    
            bbox = np.array(d['bbox'][0])
            corners = np.array(d['corners'])
            center = np.mean(corners, axis=1)
            category = d['category_name']  # Replace with your way of getting the category
            token = d['instance_token'] # Assuming 'instance_token' contains the token
 
            annotation_text = f"{category}\n{token}"

            def draw_rect(selected_corners, color):
                prev = selected_corners[-1]
                for corner in selected_corners:
                    axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                    prev = corner

            # Draw the sides
            for i in range(4):
                axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                          [corners.T[i][1], corners.T[i + 4][1]],
                          color=colors[2], linewidth=linewidth)

            # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
            draw_rect(corners.T[:4], colors[0])
            draw_rect(corners.T[4:], colors[1])

            # Draw line indicating the front
            center_bottom_forward = np.mean(corners.T[2:4], axis=0)
            center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
            axis.text(center_bottom[0], center_bottom[1], annotation_text, color='black', fontsize=1, ha='center')

            axis.plot([center_bottom[0], center_bottom_forward[0]],
                      [center_bottom[1], center_bottom_forward[1]],
                      color=colors[0], linewidth=linewidth)
            axis.plot([egobox[0], egobox[1]], [egobox[2], egobox[3]], color='r', linewidth=linewidth)
            
        fig1.savefig('lidar_projections/x'+str(cp)+'.png', dpi=600)
        

def getBBoxFromSensor(sensor, s, nusc):
    if sensor == "LIDAR_TOP":

        return getLidarBBox_of_Scenes(s, nusc)
    else:
        return BBox_from_Cameras(sensor, s, nusc)

