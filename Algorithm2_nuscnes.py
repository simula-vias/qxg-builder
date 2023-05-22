#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:07:19 2023

@author: nassim
"""

import pickle

from nuscenes.nuscenes import NuScenes
from common import *
from Nuscenes_parser import *
import argparse

import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["individual", "temporal"],
        help="Build individual graphs per frame or grow a temporal graph",
    )
    parser.add_argument("-v", "--version", default="v1.0-mini", help="nuScenes version")
    parser.add_argument(
        "-d",
        "--dataroot",
        default="./data/sets/nuscenes",
        help="Path of nuScenes dataset",
    )
    parser.add_argument(
        "-s",
        "--sensor",
        default="LIDAR_TOP",
        choices=[
            "LIDAR_TOP",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
            "CAM_BACK_LEFT",
        ],
        help="Sensor from which the graph is built",
    )
    args = parser.parse_args()

    mode = args.mode
    version = args.version
    dataroot = args.dataroot
    sensor = args.sensor

    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    scenes = nusc.scene
    path = "results/" + sensor
    os.makedirs(path, exist_ok=True)

    with open(f"results/{sensor}/times_{mode}.csv", "w") as times_file:
        times_file.write(f"scene;frame;bruteforce;smartforce;nodes;edges\n")

        for nb, s in enumerate(scenes):
            print(f"scene {nb+1}/{len(scenes)}")
            name = s["name"]

            frames = getBBoxFromSensor(sensor, s, nusc)
            res_bf = {}
            res_sf = {}

            for cp, f in enumerate(frames):
                boxes = {}
                for o in f:
                    boxes[o["instance_token"]] = Build_Rectangle(o["bbox"][0])

                if len(boxes) > 0:
                    boxes["ego"] = Build_Rectangle(o["ego_box"])
                else:
                    print(f"scene {nb}: no objects in frame {cp} for sensor {sensor}")

                grow_graph = mode == "temporal" and cp > 0

                res_sf[cp] = QXGBUILDER(
                    boxes, cp, dict(res_sf[cp - 1][0]) if grow_graph else {}
                )
                res_bf[cp] = BruteForce(
                    boxes, cp, dict(res_bf[cp - 1][0]) if grow_graph else {}
                )

                nodes = num_nodes(res_sf[cp][0])
                edges = len(res_sf[cp][0])

                times_file.write(
                    f"{name};{cp};{res_bf[cp][1]:.4f};{res_sf[cp][1]:.4f};{nodes};{edges}\n"
                )

            results = {}
            results[f"BruteForce_{name}"] = res_bf
            results[f"SmartForce_{name}"] = res_sf
            print(testing_network(results, "BruteForce_", "SmartForce_", name))

            pickle.dump(
                {
                    "scene": name,
                    "sensor": sensor,
                    "smartforce": res_sf,
                },
                open(f"results/{sensor}/{name}_{mode}.p", "wb"),
            )

    print("Done")
