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

def extract_object_pair_relationchains(results):
    pass


def encode_RA(RA):
    all_rels = "Bx|BIx|Dx|DIx|Ex|Fx|FIx|Mx|MIx|Ox|OIx|Sx|SIx|By|BIy|Dy|DIy|Ey|Fy|FIy|My|MIy|Oy|OIy|Sy|SIy".split(
        "|"
    )
    all_possibilities = {}
    for c in itertools.combinations(all_rels, 2):
        if "x" in c[0] and "y" in c[1]:
            all_possibilities[c]=0
    for c in itertools.combinations(RA, 2):
        if "x" in c[0] and "y" in c[1]:
            all_possibilities[c]=1
    return all_possibilities

def encode_DIR(DIR):
    all_rels = ['stationary', 'same direction','opposite direction']
    all_possibilities = {}
    for c in all_rels:
            all_possibilities[c]=0

    all_possibilities[DIR]=1
    return all_possibilities

def encode_QTC(DIR):
    all_rels = ['Moving Away', 'Moving Towards','Stationary']
    all_possibilities = {}
    for c in all_rels:
            all_possibilities[c]=0

    all_possibilities[DIR]=1
    return all_possibilities
 
def encode_STAR(STAR):
    all_rels = ['1', '2','3','4','5','6']
    all_possibilities = {}
    for c in all_rels:
            all_possibilities[c]=0
    for r in STAR :
            all_possibilities[str(r)]=1
    return all_possibilities


def measure_calculation():
    pass

def extract_relations(results):
    pass