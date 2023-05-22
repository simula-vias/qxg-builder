#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 09:59:01 2023

@author: nassim
"""

import json
from bitsets import bitset, bases
from collections import abc, OrderedDict
import time


class RA_Algebra:
    def __init__(self, network_dict=None):
        with open("RA_Algebra.json", "r") as f:
            self.algebra_dict = json.load(f)

        self.name = self.algebra_dict["Name"]

        if "Description" in self.algebra_dict:
            self.description = self.algebra_dict["Description"]
        else:
            self.description = "No description provided."

        self.rel_info_dict = self.algebra_dict["Relations"]

        self.elements_bitset = bitset("relset", tuple(self.rel_info_dict.keys()))

        self.elements = self.elements_bitset.supremum

        # Setup the transitivity (or composition) table to be used by Relation Set composition.
        # This code can read both the original transitivity table format and the newer compact
        # transitivity table format, which is now the default.
        self.transitivity_table = dict()
        tabledefs = self.algebra_dict["TransTable"]
        for rel1 in tabledefs:
            self.transitivity_table[rel1] = dict()
            for rel2 in tabledefs[rel1]:
                table_entry = tabledefs[rel1][rel2]
                if type(table_entry) == list:
                    entry = table_entry
                elif type(table_entry) == str:
                    if table_entry == "":
                        entry = []  # because "".split('|') = ['']
                    else:
                        entry = table_entry.split("|")
                else:
                    raise Exception("Bad entry in transitivity table")
                # print(rel1, rel2)
                self.transitivity_table[rel1][rel2] = self.elements_bitset(tuple(entry))

        self.net_dict = network_dict
        node_list = self.net_dict["nodes"]
        self.nodes = dict()
        for nd in node_list:
            self.nodes[nd[0]] = nd[1]
        self.edges = dict()
        for edge_spec in self.net_dict["edges"]:
            entry = edge_spec[2].split("|")
            constraint = self.elements_bitset(tuple(entry))
            self.edges[edge_spec[0], edge_spec[1]] = constraint
            converse = []
            for c in constraint:
                converse.append(self.converse(c))
            self.edges[edge_spec[1], edge_spec[0]] = self.elements_bitset(
                tuple(converse)
            )
        self.distances = dict()
        for dist in self.net_dict["distances"]:
            self.distances[dist[0], dist[1]] = dist[2]
            self.distances[dist[1], dist[0]] = dist[2]
        self.triangles = self.net_dict["triangles"]

    def compose(self, relset1, relset2):
        """Composition is done, element-by-element, on the cross-product
        of the two sets using the algebra's transitivity table, and
        then reducing those results to a single relation set using set
        union.
        """
        result = self.elements_bitset.infimum  # the empty relation set
        for r1 in relset1:
            for r2 in relset2:
                result = result.union(self.transitivity_table[r1][r2])
        return result

    def relset(self, relations):
        """Return a relation set (bitset) for the given relations."""
        if isinstance(relations, str):  # if relations is like 'B|M|O' or 'B', or ''
            if relations == "":
                return self.relset([])
            else:
                return self.string_to_relset(relations)
        elif isinstance(
            relations, abc.Iterable
        ):  # relations is like ['B','M','O'], ('B',), or []
            return self.elements_bitset(relations)
        else:
            raise TypeError("Input must be a string, list, tuple, or set.")

    def string_to_relset(self, st, delimiter="|"):
        """Take a string, st, like 'B|M|O' and turn it into a relation set."""
        return self.relset(st.split(delimiter))

    def converse(self, rel_or_relset):
        """Return the converse of a relation (str) or relation set (bitset).
        e.g., 'A before B' has converse 'B after A', so 'after' is the converse of 'before',
        and vice versa."""
        if isinstance(rel_or_relset, str):
            return self.rel_info_dict[rel_or_relset]["Converse"]
        else:
            return self.elements_bitset(
                (self.converse(r) for r in rel_or_relset.members())
            )

    def PC(self, verbose=False, time_limit=100000):
        """Propagate constraints in the network. Constraint propagation is a fixed-point
        iteration of a square constraint matrix.  That is, we treat the network as if it's
        a matrix, multiplying it by itself, repeatedly, until it stops changing.
        The algebra's compose method plays the role of multiplication and the RelSet +
        operation plays the role of addition in the constraint matrix multiplication.
        :param verbose: If True, then the number of iterations required is printed
        :return: True if network is consistent, otherwise False
        """
        loop_count = 0
        something_changed = True  # We'll iterate at least once
        Q = set()
        for i, ent1 in enumerate(self.nodes):
            for j, ent2 in enumerate(self.nodes):
                if i < j:
                    Q.add((ent1, ent2))
        begin = time.time()
        while len(Q) > 0:
            # Q.sort()
            ent1, ent2 = Q.pop()
            # print(ent1.name,ent2.name)
            Cij = self.edges[ent1, ent2]
            Cij_x = self.relset(
                [s for s in self.elements_bitset(tuple(Cij)) if "x" in s]
            )
            Cij_y = self.relset(
                [s for s in self.elements_bitset(tuple(Cij)) if "y" in s]
            )
            composition_x = None
            composition_y = None
            composition1_x = None
            composition1_y = None
            t1_x = None
            t1_y = None
            t2_x = None
            t2_y = None
            for k, ent3 in enumerate(self.nodes):
                if ent3 != ent1 and ent3 != ent2:
                    # if(ent1=="ego car"):
                    # print(ent1,ent2,ent3)

                    Cik = self.edges[ent1, ent3]
                    Cki = self.edges[ent3, ent1]
                    Cjk = self.edges[ent2, ent3]
                    Ckj = self.edges[ent3, ent2]

                    Cik_x = self.relset(
                        [s for s in self.elements_bitset(tuple(Cik)) if "x" in s]
                    )
                    Cik_y = self.relset(
                        [s for s in self.elements_bitset(tuple(Cik)) if "y" in s]
                    )

                    Cki_x = self.relset(
                        [s for s in self.elements_bitset(tuple(Cki)) if "x" in s]
                    )
                    Cki_y = self.relset(
                        [s for s in self.elements_bitset(tuple(Cki)) if "y" in s]
                    )

                    Cjk_x = self.relset(
                        [s for s in self.elements_bitset(tuple(Cjk)) if "x" in s]
                    )
                    Cjk_y = self.relset(
                        [s for s in self.elements_bitset(tuple(Cjk)) if "y" in s]
                    )

                    Ckj_x = self.relset(
                        [s for s in self.elements_bitset(tuple(Ckj)) if "x" in s]
                    )
                    Ckj_y = self.relset(
                        [s for s in self.elements_bitset(tuple(Ckj)) if "y" in s]
                    )

                    composition_x = self.compose(Cij_x, Cjk_x)
                    composition_y = self.compose(Cij_y, Cjk_y)

                    t1_x = Cik_x.intersection(composition_x)
                    t1_y = Cik_y.intersection(composition_y)

                    if len(t1_x) == 0:
                        if verbose:
                            print(
                                f"Propagation suspended; the network is inconsistent."
                            )
                        return False
                    if len(t1_y) == 0:
                        if verbose:
                            print(
                                f"Propagation suspended; the network is inconsistent."
                            )
                        return False
                    if t1_x != Cik_x or t1_y != Cik_y:
                        print("Pruned")

                        self.edges[ent1, ent3] = t1_x.union(t1_y)
                        self.edges[ent3, ent1] = self.converse(t1_x).union(
                            self.converse(t1_y)
                        )
                        Q.add((ent1, ent3))
                    composition1_x = self.compose(Cki_x, Cij_x)
                    composition1_y = self.compose(Cki_y, Cij_y)

                    t2_x = Ckj_x.intersection(composition1_x)
                    t2_y = Ckj_y.intersection(composition1_y)
                    if len(t2_x) == 0:
                        if verbose:
                            print(
                                f"Propagation suspended; the network is inconsistent."
                            )
                        return False
                    if len(t2_y) == 0:
                        if verbose:
                            print(
                                f"Propagation suspended; the network is inconsistent."
                            )
                        return False
                    if t2_x != Ckj_x or t2_y != Ckj_y:
                        print("Pruned")
                        self.edges[ent3, ent2] = t2_x.union(t2_y)
                        self.edges[ent2, ent3] = self.converse(t2_x).union(
                            self.converse(t2_y)
                        )
                        Q.add((ent3, ent2))
                    loop_count += 1
                    end = time.time()
                    if (end - begin) > time_limit:
                        break
            if (end - begin) > time_limit:
                break
            # If any product is empty then the Network is inconsistent

        # Update the Entity/Node classes to reflect changes due to constraint propagation

        if verbose:
            print(f"Number of iterations: {loop_count}")
        return True

    def PC_Composition(self, verbose=False, triangle=None):
        """
        Triangle composition
        """

        begin = time.time()
        ent1, ent2, ent3 = triangle 

        Cij_x_l = []
        Cij_y_l = []

        Cij = self.edges[ent1, ent2]
        for s in self.elements_bitset(tuple(Cij)):
            if "x" in s:
                Cij_x_l.append(s)
            elif "y" in s:
                Cij_y_l.append(s)

        Cij_x = self.relset(Cij_x_l)
        Cij_y = self.relset(Cij_y_l)

        Cik_x_l = []
        Cik_y_l = []

        Cik = self.edges[ent1, ent3]
        for s in self.elements_bitset(tuple(Cik)):
            if "x" in s:
                Cik_x_l.append(s)
            elif "y" in s:
                Cik_y_l.append(s)

        Cik_x = self.relset(Cik_x_l)
        Cik_y = self.relset(Cik_y_l)

        Cjk_x_l = []
        Cjk_y_l = []

        Cjk = self.edges[ent2, ent3]
        for s in self.elements_bitset(tuple(Cjk)):
            if "x" in s:
                Cjk_x_l.append(s)
            elif "y" in s:
                Cjk_y_l.append(s)

        Cjk_x = self.relset(Cjk_x_l)
        Cjk_y = self.relset(Cjk_y_l)

        t1_x = Cik_x.intersection(self.compose(Cij_x, Cjk_x))
        t1_y = Cik_y.intersection(self.compose(Cij_y, Cjk_y))

        self.edges[ent1, ent3] = t1_x.union(t1_y)
        # self.edges[ent3, ent1]= self.converse(t1_x).union(self.converse(t1_y))
        end = time.time()
        # print('Composition time:',(end-begin))

        return True

    def summary(self):
        for e in self.edges:
            if e[0] < e[1]:
                constraint_x = " v ".join([r for r in self.edges[e] if "x" in r])
                constraint_y = " v ".join([r for r in self.edges[e] if "y" in r])
                if len(constraint_x) > 0:
                    print(
                        str(e[0])
                        + " --- (("
                        + str(constraint_x)
                        + ") , ("
                        + str(constraint_y)
                        + ")) --- "
                        + str(e[1])
                    )
                else:
                    print(
                        str(e[0]) + " --- (" + str(self.edges[e]) + ") --- " + str(e[1])
                    )

    # TODO: Don't hard code the legend below; make it depend on an abbreviations file (JSON)

    def Translate(self):
        Translation = {
            "Dx|Sx|Fx|Ex|Dy|Sy|Fy|Ey": "B",
            "Dx|Sx|Fx|Ex|My|By": "S",
            "Dx|Sx|Fx|Ex|MIy|BIy": "N",
            "MIx|BIx|Dy|Sy|Fy|Ey": "E",
            "Mx|Bx|Dy|Sy|Fy|Ey": "W",
            "MIx|BIx|MIy|BIy": "NE",
            "Mx|Bx|MIy|BIy": "NW",
            "MIx|BIx|My|By": "SE",
            "Mx|Bx|My|By": "SW",
            "FIx|Ox|My|By": "S:SW",
            "SIx|OIx|My|By": "S:SE",
            "FIx|Ox|MIy|BIy": "N:NW",
            "SIx|OIx|MIy|BIy": "N:NE",
            "FIx|Ox|Dy|Sy|Fy|Ey": "B:W",
            "SIx|OIx|Dy|Sy|Fy|Ey": "B:E",
            "Dx|Sx|Fx|Ex|FIy|Oy": "B:S",
            "Dx|Sx|Fx|Ex|SIy|OIy": "B:N",
            "Mx|Bx|FIy|Oy": "W:SW",
            "Mx|Bx|SIy|OIy": "W:NW",
            "MIx|BIx|FIy|Oy": "E:SE",
            "MIx|BIx|SIy|OIy": "E:NE",
            "DIx|My|By": "S:SW:SE",
            "DIx|MIy|BIy": "N:NW:NE",
            "DIx|Dy|Sy|Fy|Ey": "B:W:E",
            "Dx|Sx|Fx|Ex|DIy": "B:N:S",
            "Mx|Bx|DIy": "W:NW:SW",
            "MIx|BIx|DIy": "E:NE:SE",
            "Ox|FIx|Oy|FIy": "B:S:SW:W",
            "Ox|FIx|SIy|OIy": "B:W:NW:N",
            "SIx|OIx|Oy|FIy": "B:S:E:SE",
            "SIx|OIx|SIy|OIy": "B:N:NE:E",
            "Ox|FIx|DIy": "B:S:SW:W:NW:N",
            "SIx|OIx|DIy": "B:S:SE:E:NE:N",
            "DIx|FIy|Oy": "B:S:SW:W:E:SE",
            "DIx|SIy|OIy": "B:W:NW:N:NE:E",
            "DIx|DIy": "B:S:SW:W:NW:N:NE:E:SE",
        }

        for i, ent1 in enumerate(self.nodes):
            for j, ent2 in enumerate(self.nodes):
                if ent1 != ent2:
                    Cij = self.edges[ent1, ent2]
                    for t in Translation:
                        Cij = self.elements_bitset(tuple(Cij))
                        t1 = self.relset(t).intersection(Cij)
                        if len(t1) == len(Cij):
                            self.edges[ent1, ent2] = Translation[t]
