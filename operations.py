# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:32:56 2022

@author: Nassim
"""

from math import hypot
import numpy as np

import numpy as np
import matplotlib.pyplot as plt


def distance(p1, p2):
    """Euclidean distance between two points."""
    x1, y1 = p1
    x2, y2 = p2
    return hypot(x2 - x1, y2 - y1)


def compute_RA_Algebra(bb1, bb2, r):
   
    if len(bb1) == 4 and len(bb2) == 4:  # 2D version
        if "x" in r:
            X = compute_Allen_relation((bb1[0], bb1[2]), (bb2[0], bb2[2]), r, "x")
            return X
        if "y" in r:
            Y = compute_Allen_relation((bb1[1], bb1[3]), (bb2[1], bb2[3]), r, "y")
            return Y
  
    else:
        raise ValueError("bb1 and bb2 must have length of 4 (2D) or 6 (3D)")


def compute_Allen_relation(i1, i2, r, axis=None):
    if "I" in r:

        return (
            ("BI" in r and i2[1] < i1[0])
            or ("MI" in r and i2[1] == i1[0])
            or ("OI" in r and i2[0] < i1[0] < i2[1] and i1[0] < i2[1] < i1[1])
            or ("SI" in r and i2[0] == i1[0] and i2[1] < i1[1])
            or ("DI" in r and i1[0] < i2[0] < i1[1] and i1[0] < i2[1] < i1[1])
            or ("FI" in r and i1[0] < i2[0] < i1[1] and i2[1] == i1[1])
            or ("E" in r and i2[0] == i1[0] and i2[1] == i1[1])
        )

    else:
       
        return (
            ("B" in r and i1[1] < i2[0])
            or ("M" in r and i1[1] == i2[0])
            or ("O" in r and  i1[0] < i2[0] < i1[1] and i2[0] < i1[1] < i2[1])
            or ("S" in r and i1[0] == i2[0] and i1[1] < i2[1])
            or ("D" in r and i2[0] < i1[0] < i2[1] and i2[0] < i1[1] < i2[1])
            or ("F" in r and i2[0] < i1[0] < i2[1] and i1[1] == i2[1])
            or ("E" in r and i1[0] == i2[0] and i1[1] == i2[1])
        )


