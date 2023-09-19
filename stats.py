#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:34:27 2023

@author: nassim
"""

import pandas as pd

import copy
from scipy.spatial.distance import pdist, squareform

import pickle

from nuscenes.nuscenes import NuScenes
from common import *
from Nuscenes_parser import *
from Action_Extraction import *
import argparse
import os
from demo import *
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


def train(final_dataset_train,nb_estimators):


    object_pair_groups=[]
    for i, o in final_dataset_train.groupby(['action']):
    
        object_pair_groups.append(o)

    trees={}
    for last_n_seconds in object_pair_groups:
            X=last_n_seconds[last_n_seconds.columns.difference(['object_pair','frameidx','scene','action','Object_1','Object_2'])]
            X=X.dropna(axis=0)
        
            features = X.columns
            
            clf = IsolationForest(n_estimators=nb_estimators,random_state=round(time.time()))
            clf.fit(X)
            trees[last_n_seconds['action'].values[0]]=clf
    return trees

def test(final_dataset_test,trees,anomaly,grps,action):
    object_pair_groups=[]
    for i, o in final_dataset_test.groupby(['action']):
        if o.iloc[0]['action']==action:
            object_pair_groups.append(o)
    
    for last_n_seconds in object_pair_groups:
        
            X=last_n_seconds[last_n_seconds.columns.difference(['object_pair','frameidx','scene','action','Object_1','Object_2'])]
            X=X.dropna(axis=0)
            features=X.columns
            try:
                clf=trees[action]
                predictions = clf.predict(X)
                for idx,prediction in enumerate(predictions):
                    if prediction == -1:
                        print("The test point is an anomaly.")
                        #paths1=  all_paths_isolation_forest(clf,features,datapoint=X.iloc[idx])
                        rules = get_data_point_rules(clf, X.iloc[idx],features)
                        mask = last_n_seconds.iloc[idx].to_frame().T == 1

                        # Use the all() method to check if all elements in each column are equal to 1
                        columns_with_only_1 = mask.any()

                        selected_columns = last_n_seconds.iloc[idx].to_frame().T.columns[columns_with_only_1]
                        # Use the any() method to check if any element in each column is equal to 1
                        anomaly[(last_n_seconds.iloc[idx]['Object_1'],last_n_seconds.iloc[idx]['Object_2']),last_n_seconds.iloc[idx]['frameidx'],action]=last_n_seconds.iloc[idx].to_frame().T #[set(rules),selected_columns]#,last_n_seconds.iloc[idx]['frameidx'],last_n_seconds.iloc[idx]['action'],last_n_seconds.iloc[idx]['scene']]=list(selected_columns)#[set(rules),selected_columns]
                    else:
                        #paths=  all_paths_isolation_forest(clf,features,datapoint=X.iloc[idx])
                        rules = get_data_point_rules(clf, X.iloc[idx],features)
                        
                        mask = last_n_seconds.iloc[idx].to_frame().T == 1

                        # Use the all() method to check if all elements in each column are equal to 1
                        columns_with_only_1 = mask.any()

                        selected_columns = last_n_seconds.iloc[idx].to_frame().T.columns[columns_with_only_1]
                        # Use the any() method to check if any element in each column is equal to 1
                        grps[(last_n_seconds.iloc[idx]['Object_1'],last_n_seconds.iloc[idx]['Object_2']),last_n_seconds.iloc[idx]['frameidx'],action]=last_n_seconds.iloc[idx].to_frame().T #[set(rules),selected_columns]#,last_n_seconds.iloc[idx]['frameidx'],last_n_seconds.iloc[idx]['action'],last_n_seconds.iloc[idx]['scene']]=list(selected_columns)#[set(rules),selected_columns]
            except Exception as  e :
                print(e)
                print(last_n_seconds['object_pair'].values[0],last_n_seconds['action'].values[0],'was never seen before')


def extract_rules(estimator, data_point, feature_names):
    """Extract the rules from a single tree using feature names and keep only those where the feature value is 1."""
    rules = []
    tree = estimator.tree_
    feature = tree.feature
    threshold = tree.threshold
    node = 0  # Start from the root

    while feature[node] != -2:  # -2 indicates a leaf node in Scikit-learn's tree structure
            feature_name = feature_names[feature[node]]
            if data_point[feature[node]] < threshold[node]:
                rules.append(f"{feature_name} < {threshold[node]}")
                node = tree.children_left[node]
            else:
                rules.append(f"{feature_name}>={threshold[node]}")
                node = tree.children_right[node]
            #if data_point[feature[node]] < threshold[node]:
               # node = tree.children_left[node]
            #else:
                #node = tree.children_right[node]

    return rules


def get_data_point_rules(model, data_point,features):
    """Extract rules from all trees in the Isolation Forest."""
    all_rules = []
    for estimator in model.estimators_:
        rules = extract_rules(estimator,data_point,features)
        all_rules.extend(rules)

    # Aggregate rules for interpretability
    aggregated_rules = {rule: all_rules.count(rule) for rule in set(all_rules)}
    return set(aggregated_rules)



mode = 'temporal'
version = "v1.0-mini"
dataroot = "/home/nassim/Desktop/Self-DrivingCara-XAI/data/sets/nuscenes"
sensor = 'LIDAR_TOP'



nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

scenes = nusc.scene
results = {}

launch(scenes,nusc,sensor,mode,results)

dfs=[]
encoder = LabelEncoder()

for k,v in results.items():
    labels=v[1]
    elements=[]
    for f in v[0]:
        for o in v[0][f][1]:
            e=copy.copy(o)
            e.append(k)
            try:
                e.append(v[1][o[1]])
            except:
                e.append('cruising')
            elements.append(e)

    df=pd.DataFrame(elements,columns=['object_pair','frameidx','distance','ra','direction','strar_o1','strar_o2','scene','action'])
    
    dfs.append(df)


dataset=pd.concat(dfs)
#dataset = pd.get_dummies(dataset, columns=['distance','ra','direction','strar_o1','strar_o2'],prefix_sep=[' ',' ',' ',' ',' '])
all_categories2 = dataset['distance'].unique()
all_categories3 = dataset['ra'].unique()
all_categories4 = dataset['direction'].unique()

category_mapping2 = {category: i for i, category in enumerate(all_categories2)}
category_mapping3 = {category: i for i, category in enumerate(all_categories3)}
category_mapping4 = {category: i for i, category in enumerate(all_categories4)}


dataset['distance'] = dataset['distance'].replace(category_mapping2)
dataset['ra'] = dataset['ra'].replace(category_mapping3)
dataset['direction'] = dataset['direction'].replace(category_mapping4)

scene_groups=dataset.groupby(['scene'])
scenes=[]
for idx, s in scene_groups:
    
    scenes.append(s)




train_dataset=pd.concat(scenes[0:7])

test_dataset=pd.concat(scenes[7:10])
 

groups_train=train_dataset.groupby(['object_pair','scene','action'])
groups_test=test_dataset.groupby(['object_pair','scene','action'])


final_dataset_train=[]
final_dataset_test=[]

n=5
for i,g in groups_train :
    last_n_seconds=g.tail(n)


    if len(last_n_seconds)>=n:# and last_n_seconds['object_pair'].values[0][1]=='ego':

        X=last_n_seconds[last_n_seconds.columns.difference(['object_pair','frameidx','scene','action','scene'])]
        flattened_data = X.values.flatten()

        X = pd.DataFrame([flattened_data], columns=[f'{i}_sec{j+1}' for j in range(X.shape[0]) for i in X.columns])
        X['object_pair']=[(last_n_seconds['object_pair'].values[0][0].split('_')[0],last_n_seconds['object_pair'].values[0][1].split('_')[0])]
        X['Object_1']=[last_n_seconds['object_pair'].values[0][0]]
        X['Object_2']=[last_n_seconds['object_pair'].values[0][1]]


        X['action']=[last_n_seconds['action'].values[0]]
        X['scene']=[last_n_seconds['scene'].values[0]]
        X['frameidx']=[last_n_seconds['frameidx'].values[0]]

        final_dataset_train.append(X)


for i,g in groups_test :
    last_n_seconds=g.tail(n)


    if len(last_n_seconds)>=n : #and last_n_seconds['object_pair'].values[0][1]=='ego':

        X=last_n_seconds[last_n_seconds.columns.difference(['object_pair','frameidx','scene','action','scene'])]
        flattened_data = X.values.flatten()

        X = pd.DataFrame([flattened_data], columns=[f'{i}_sec{j+1}' for j in range(X.shape[0]) for i in X.columns])
        X['object_pair']=[(last_n_seconds['object_pair'].values[0][0].split('_')[0],last_n_seconds['object_pair'].values[0][1].split('_')[0])]
        X['Object_1']=[last_n_seconds['object_pair'].values[0][0]]
        X['Object_2']=[last_n_seconds['object_pair'].values[0][1]]


        X['action']=[last_n_seconds['action'].values[0]]
        X['scene']=[last_n_seconds['scene'].values[0]]
        X['frameidx']=[last_n_seconds['frameidx'].values[len(last_n_seconds)-1]]

        final_dataset_test.append(X)


# Combine unique values from both dataframes

temp_train=pd.concat(final_dataset_train)
temp_test=pd.concat(final_dataset_test)
temp_train['object_pair']=temp_train['object_pair'].astype(str)
temp_test['object_pair']=temp_test['object_pair'].astype(str)


all_categories1 = pd.concat([temp_train['object_pair'], temp_test['object_pair']]).unique()

# Create a mapping from category to integer
category_mapping1 = {category: i for i, category in enumerate(all_categories1)}

temp_train['object_pair'] = temp_train['object_pair'].replace(category_mapping1)
temp_test['object_pair'] = temp_test['object_pair'].replace(category_mapping1)


grps={}
grps1={}
anomaly={}
anomaly1={}

trees=train(temp_train,100)

test(temp_test, trees,anomaly,grps,'Stop')
#test(temp_test, trees,anomaly1,grps1,'Accelerate')


from collections import defaultdict
from itertools import combinations
K = defaultdict(set)
for key in anomaly.keys():
    K[key[0]].add(key)
    K[key[1]].add(key)

# Step 2 and 3: Find triangles
triplets = {}
for tuples in combinations(K, 3):
    i,j,k=sorted(tuples,reverse=True)
    x=None
    y=None
    z=None

    if (i,j) in anomaly:
        x=[(i,j),anomaly[(i,j)]]
    if (j,i) in anomaly:
        x=[(j,i),anomaly[(j,i)]]
    if (i,k) in anomaly: 
        y=[(i,k),anomaly[(i,k)]]
    if (k,i) in anomaly: 
        y=[ (k,i),anomaly[(k,i)]]
    if (j,k) in anomaly: 
        z=[(j,k),anomaly[(j,k)]]
    if (k,j) in anomaly: 
         z=[(k,j),anomaly[(k,j)]]
    my_list = [var for var in [x, y, z] if var is not None]
    if my_list:
        triplets[(i,j,k)]=my_list
# Print the triangles
for triangle in triangles:
    print(triangle)





scenes = nusc.scene

for nb, s in enumerate(scenes):
    name = s["name"]
    if name=='scene-1100':
        print(f"scene {nb+1}/{len(scenes)}")
        
    
        frames = getBBoxFromSensor(sensor, s, nusc)
        print_boxes(frames)


