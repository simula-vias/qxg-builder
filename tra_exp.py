#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:10:04 2023

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
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score


from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import time

def train(final_dataset_train, nb_estimators):
    object_pair_groups = [o for _, o in final_dataset_train.groupby(['action'])]
    
    # Dictionaries to store different types of classifiers
    isolation_forest_trees = {}
    one_class_svm_models = {}
    lof_models = {}
    elliptic_envelope_models = {}
    feature_order = None  # Initialize variable to store feature order

    for last_n_seconds in object_pair_groups:
        X = last_n_seconds.drop(['object_pair', 'frameidx', 'scene', 'action', 'Object_1', 'Object_2'], axis=1).dropna()
        if feature_order is None:
            feature_order = X.columns.tolist()  # Save the feature order during the first iteration
        
        # Train Isolation Forest
        isolation_forest_clf = IsolationForest(n_estimators=nb_estimators, random_state=round(time.time()))
        isolation_forest_clf.fit(X)
        isolation_forest_trees[last_n_seconds['action'].values[0]] = isolation_forest_clf
        
        # Train One-Class SVM
        one_class_svm_clf = OneClassSVM()
        one_class_svm_clf.fit(X)
        one_class_svm_models[last_n_seconds['action'].values[0]] = one_class_svm_clf
        
        # Train Local Outlier Factor
        lof_clf = LocalOutlierFactor(novelty=True)
        lof_clf.fit(X)
        lof_models[last_n_seconds['action'].values[0]] = lof_clf
        
        # Train Elliptic Envelope
        elliptic_envelope_clf = EllipticEnvelope()
        elliptic_envelope_clf.fit(X)
        elliptic_envelope_models[last_n_seconds['action'].values[0]] = elliptic_envelope_clf
    
    # You can return these as separate dictionaries or nest them into a single dictionary
    return isolation_forest_trees, one_class_svm_models, lof_models, elliptic_envelope_models,feature_order


def test(final_dataset_test,feature_order,trees,action):
    object_pair_groups=[]
    metrics = {'accuracy': [],'precision': [], 'recall': [], 'f1_score': []}

    for i, o in final_dataset_test.groupby(['action']):
        if o.iloc[0]['action']==action:
            object_pair_groups.append(o)
    
    for last_n_seconds in object_pair_groups:
        
            X=last_n_seconds[last_n_seconds.columns.difference(['object_pair','frameidx','scene','action','Object_1','Object_2'])]
            X = X[feature_order]
            X=X.dropna(axis=0)
            features=X.columns

           
            last_n_seconds['action'] = 1

            y_true=last_n_seconds['action'].values
            try:
                clf=trees[action]
                y_pred = clf.predict(X)
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                metrics['accuracy'].append(accuracy)
                metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['f1_score'].append(f1)
            except Exception as  e :
                print(e)
    print(metrics)
    return metrics




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
dataset.to_csv('whole_dataset.csv')
scene_groups=dataset.groupby(['scene'])
by_scenes=[]
for idx, s in scene_groups:
    
    by_scenes.append(s)


cut=len(by_scenes)-round((30*len(by_scenes))/100)

train_dataset=pd.concat(by_scenes[0:cut])

test_dataset=pd.concat(by_scenes[cut:10])

 

groups_train=train_dataset.groupby(['object_pair','scene'])
groups_test=test_dataset.groupby(['object_pair','scene'])


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




isolation_forest_trees, one_class_svm_models, lof_models, elliptic_envelope_models,feature_order=train(temp_train,10)




test(temp_test,feature_order, isolation_forest_trees,'cruising')
test(temp_test,feature_order, isolation_forest_trees,'Accelerate')
test(temp_test,feature_order, isolation_forest_trees,'Stop')



test(temp_test,feature_order, one_class_svm_models,'cruising')
test(temp_test,feature_order, one_class_svm_models,'Accelerate')
test(temp_test,feature_order, one_class_svm_models,'Stop')


test(temp_test,feature_order, lof_models,'cruising')
test(temp_test,feature_order, lof_models,'Accelerate')
test(temp_test,feature_order, lof_models,'Stop')


test(temp_test,feature_order, elliptic_envelope_models,'cruising')
test(temp_test,feature_order, elliptic_envelope_models,'Accelerate')
test(temp_test,feature_order, elliptic_envelope_models,'Stop')

