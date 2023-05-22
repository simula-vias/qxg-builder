import itertools
import json
import time
import vetqca_ops
import numpy as np
from RA import RA_Algebra




def testing_network(results, prefix1, prefix2, scene):
    for frame in results[prefix1 + str(scene)].keys():
        for key in results[prefix1 + str(scene)][frame][0].keys():
            l1 = set(results[prefix1 + str(scene)][frame][0][key])
            l2 = set(results[prefix2 + str(scene)][frame][0][key])

            if l1 != l2:
                return False

    return True

def get_spatial_relations(boxes, o_i, o_j, rels):
    learned = set()
    bb1 = list(boxes[o_i])
    bb2 = list(boxes[o_j])

    for r in rels:
        if isinstance(r, tuple):
            for r1 in list(r):
                answer = vetqca_ops.compute_RA_Algebra(bb1, bb2, r1)
                if answer:
                    learned.add(r1)
        else:
            answer = vetqca_ops.compute_RA_Algebra(bb1, bb2, r1)
            if answer:
                learned.add(r)
    return learned



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


def QXGBUILDER(boxes, frame_idx=0, initial_graph={}):
    begin = time.time()
    learned = initial_graph
    for o_i, o_j in itertools.combinations(boxes, 2):
        rels = learned.get((o_i, o_j), [])
        rels.append((frame_idx, get_relation(boxes[o_i], boxes[o_j])))
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
                answer = vetqca_ops.compute_RA_Algebra(bb1, bb2, r1)
                if answer:
                    learned.add(r1)
        else:
            answer = vetqca_ops.compute_RA_Algebra(bb1, bb2, r1)
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
