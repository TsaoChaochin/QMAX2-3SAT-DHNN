import math
import random
from itertools import permutations
import numpy as np
import time

import pandas as pd

start = time.perf_counter()  # calculate time
# Some performance evaluation metrics
MSETotal = RMSETotal = MAETotal = SSETotal = MBETotal = MAPETotal = SMAPETotal = SBCTotal = 0.0
RMSEEnergy = MAEEnergy = SSEEnergy = MAPEEnergy = 0
MSETest = RMSETest = MAETest = SSETest = MBETest = MAPETest = SMAPETest = 0
TOL = 0.001  # Tolerance Value
weight_4SAT = [
    [0.0625, 0.0625, 0.0625, 0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, 0.03125, 0.03125, 0.03125,
     0.03125, -1 / 96],
    [0.0625, 0.0625, 0.0625, -0.0625, -0.0625, -0.0625, 0.0625, -0.0625, 0.0625, 0.0625, 0.03125, -0.03125, -0.03125,
     -0.03125, 1 / 96],
    [0.0625, 0.0625, -0.0625, 0.0625, -0.0625, 0.0625, -0.0625, 0.0625, -0.0625, 0.0625, -0.03125, 0.03125, -0.03125,
     -0.03125, 1 / 96],
    [0.0625, 0.0625, -0.0625, -0.0625, -0.0625, 0.0625, 0.0625, 0.0625, 0.0625, -0.0625, -0.03125, -0.03125, 0.03125,
     0.03125, -1 / 96],
    [0.0625, -0.0625, 0.0625, 0.0625, 0.0625, -0.0625, -0.0625, 0.0625, 0.0625, -0.0625, -0.03125, -0.03125, 0.03125,
     -0.03125, 1 / 96],
    [0.0625, -0.0625, 0.0625, -0.0625, 0.0625, -0.0625, 0.0625, 0.0625, -0.0625, 0.0625, -0.03125, 0.03125, -0.03125,
     0.03125, -1 / 96],
    [0.0625, -0.0625, -0.0625, 0.0625, 0.0625, 0.0625, -0.0625, -0.0625, 0.0625, 0.0625, 0.03125, -0.03125, -0.03125,
     0.03125, -1 / 96],
    [0.0625, -0.0625, -0.0625, -0.0625, 0.0625, 0.0625, 0.0625, -0.0625, -0.0625, -0.0625, 0.03125, 0.03125, 0.03125,
     -0.03125, 1 / 96],
    [-0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, -0.0625, -0.0625, -0.0625, -0.03125, -0.03125, -0.03125,
     0.03125, 1 / 96],
    [-0.0625, 0.0625, 0.0625, -0.0625, 0.0625, 0.0625, -0.0625, -0.0625, 0.0625, 0.0625, -0.03125, 0.03125, 0.03125,
     -0.03125, -1 / 96],
    [-0.0625, 0.0625, -0.0625, 0.0625, 0.0625, -0.0625, 0.0625, 0.0625, -0.0625, 0.0625, 0.03125, -0.03125, 0.03125,
     -0.03125, -1 / 96],
    [-0.0625, 0.0625, -0.0625, -0.0625, 0.0625, -0.0625, -0.0625, 0.0625, 0.0625, -0.0625, 0.03125, 0.03125, -0.03125,
     0.03125, 1 / 96],
    [-0.0625, -0.0625, 0.0625, 0.0625, -0.0625, 0.0625, 0.0625, 0.0625, 0.0625, -0.0625, 0.03125, 0.03125, -0.03125,
     -0.03125, -1 / 96],
    [-0.0625, -0.0625, 0.0625, -0.0625, -0.0625, 0.0625, -0.0625, 0.0625, -0.0625, 0.0625, 0.03125, -0.03125, 0.03125,
     0.03125, 1 / 96],
    [-0.0625, -0.0625, -0.0625, 0.0625, -0.0625, -0.0625, 0.0625, -0.0625, 0.0625, 0.0625, -0.03125, 0.03125, 0.03125,
     0.03125, 1 / 96],
    [-0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.0625, -0.03125, -0.03125,
     -0.03125, -0.03125, -1 / 96]
]
weight_3SAT = [
    [0.125, 0.125, 0.125, -0.125, -0.125, -0.125, 0.0625],
    [0.125, 0.125, -0.125, -0.125, 0.125, 0.125, -0.0625],
    [0.125, -0.125, 0.125, 0.125, -0.125, 0.125, -0.0625],
    [0.125, -0.125, -0.125, 0.125, 0.125, -0.125, 0.0625],
    [-0.125, 0.125, 0.125, 0.125, 0.125, -0.125, -0.0625],
    [-0.125, 0.125, -0.125, 0.125, -0.125, 0.125, 0.0625],
    [-0.125, -0.125, 0.125, -0.125, 0.125, 0.125, 0.0625],
    [-0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.0625]
]
weight_2SAT = [
    [0.25, 0.25, -0.25],
    [0.25, -0.25, 0.25],
    [-0.25, 0.25, 0.25],
    [-0.25, -0.25, -0.25]
]
weight_1SAT = [
    [0.5],
    [-0.5]
]
weight_3RedSAT = [
    [0.25, 0.25, 0, -0.25, 0, 0],  # 0 : (A V B V C) ∧ (A V B V _C) = A V B
    [0.25, 0, 0.25, 0, -0.25, 0],  # 1 : (A V B V C) ∧ (A V _B V C) = A V C
    [0, 0.25, 0.25, 0, 0, -0.25],  # 2 : (A V B V C) ∧ (_A V B V C) = B V C
    [-0.25, -0.25, 0, -0.25, 0, 0],  # 3 : (_A V _B V _C) ∧ (_A V _B V C) = _A V _B
    [-0.25, 0, -0.25, 0, -0.25, 0],  # 4 : (_A V _B V _C) ∧ (_A V B V _C) = _A V _C
    [0, -0.25, -0.25, 0, 0, -0.25],  # 5 : (_A V _B V _C) ∧ (A V _B V _C) = _B V _C
    [0.25, -0.25, 0, 0.25, 0, 0],  # 6 : (A V _B V _C) ∧ (A V _B V C) = A V _B
    [0.25, 0, -0.25, 0, 0.25, 0],  # 7 : (A V _B V _C) ∧ (A V B V _C) = A V _C
    [-0.25, 0.25, 0, 0.25, 0, 0],  # 8 : (_A V B V C) ∧ (_A V B V _C) = _A V B
    [0, 0.25, -0.25, 0, 0, 0.25],  # 9 : (A V B V _C) ∧ (_A V B V _C) = B V _C
    [-0.25, 0, 0.25, 0, 0.25, 0],  # 10 : (_A V B V C) ∧ (_A V _B V C) = _A V C
    [0, -0.25, 0.25, 0, 0, 0.25]  # 11 : (A V _B V C) ∧ (_A V _B V C) = _B V C
]
weight_2RedSAT = [
    [0.5, 0],  # 0 : (A V B) ∧ (A V _B) = A
    [0, 0.5],  # 1 : (A V B) ∧ (_A V B) = B
    [-0.5, 0],  # 2 : (_A V _B) ∧ (_A V B) = _A
    [0, -0.5]  # 3 : (_A V _B) ∧ (A V _B) = _B
]
weight_MAX2SAT = [
    [0, 0, 0.25, 0.25, -0.25],  # 0 : (_p V _q) ∧ (_p V q) ∧ (p V _q) ∧ (p V _r) ∧ (q V r)
    [0, 0, 0.25, -0.25, 0.25],  # 1 : (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (q V _r) ∧ (p V r)
    [0, 0, 0.25, -0.25, 0.25],  # 2: (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_p V _r) ∧ (_q V r)
    [0, 0, 0.25, 0.25, -0.25]  # 3 : (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_q V _r) ∧ (_p V r)
]


# PART1 Related Function
# Generate clauses of different orders based on the number of neurons
# SAT_type = 0: ZRAN2,3SAT
# SAT_type = 'MAX2SAT': ZRAN2,3SAT + MAX2SAT
# SAT_type = 1: JRAN2,3SAT
# SAT_type = 2: BRAN2SAT
# SAT_type = 3: YRAN2SAT
def Combination(n, SAT_type):
    solutions = []
    a, b, c, d, e, f, g = [0] * 7
    if SAT_type in (0, 0.2, 0.4, 0.6, 0.8, 0.99):
        for d in range(0, n // 4 + 1):  # 4-SAT
            r1 = n - 4 * d
            for c in range(0, r1 // 3 + 1):  # 3-SAT
                r2 = r1 - 3 * c
                for e in range(0, r2 // 3 + 1):  # redundant 3-SAT
                    r3 = r2 - 3 * e
                    if r3 >= 0 and r3 % 2 == 0:
                        b = r3 // 2  # 2-SAT
                        solutions.append((a, b, c, d, e, f, g))
        return solutions
    elif SAT_type in ('MAX2SAT', 'MAX2SAT0.2', 'MAX2SAT0.4', 'MAX2SAT0.6', 'MAX2SAT0.8', 'MAX2SAT1'):
        for d in range(0, n // 4 + 1):  # 4-SAT
            r1 = n - 4 * d
            for c in range(0, r1 // 3 + 1):  # 3-SAT
                r2 = r1 - 3 * c
                for e in range(0, r2 // 3 + 1):  # redundant 3-SAT
                    r3 = r2 - 3 * e
                    for g in range(0, r3 // 3 + 1):  # MAX2SAT
                        r4 = r3 - 3 * g
                        if r4 >= 0 and r4 % 2 == 0:
                            b = r4 // 2  # 2-SAT
                            solutions.append((a, b, c, d, e, f, g))
        return solutions
    elif SAT_type == 1:
        for c in range(0, n // 3 + 1):  # 3SAT
            r1 = n - 3 * c
            if r1 >= 0 and r1 % 2 == 0:
                b = r1 // 2  # 2-SAT
                solutions.append((a, b, c, d, e, f, g))
        return solutions
    elif SAT_type == 2:
        for b in range(0, n // 2 + 1):  # 2-SAT
            r1 = n - 2 * b
            for f in range(0, r1 // 2 + 1):  # redundant 2-SAT
                r2 = r1 - 2 * f
                a = r2  # 1-SAT
                solutions.append((a, b, c, d, e, f, g))
        return solutions
    elif SAT_type == 3:
        for b in range(0, n // 2 + 1):  # 2-SAT
            r1 = n - 2 * b
            a = r1  # 1-SAT
            solutions.append((a, b, c, d, e, f, g))
        return solutions


def simple_random_solution(solutions, SAT_type):
    ratio_range_4SAT = {
        0: (0.0, 1.0),
        0.2: (0.0, 0.2),
        0.4: (0.2, 0.4),
        0.6: (0.4, 0.6),
        0.8: (0.6, 0.8),
        0.99: (0.8, 1.0)
    }
    ratio_range_MAX2SAT = {
        'MAX2SAT': (0.0, 1.0),
        'MAX2SAT0.2': (0.0, 0.2),
        'MAX2SAT0.4': (0.2, 0.4),
        'MAX2SAT0.6': (0.4, 0.6),
        'MAX2SAT0.8': (0.6, 0.8),
        'MAX2SAT1': (0.8, 1.0)
    }
    filtered = []
    a, b, c, d, e, f, g = [0] * 7
    if SAT_type in (0, 0.2, 0.4, 0.6, 0.8, 0.99):
        low, high = ratio_range_4SAT[SAT_type]
        for a, b, c, d, e, f, g in solutions:
            total = a + b + c + d + e + f + g
            ratio = d / total
            if low <= ratio <= high:
                filtered.append((a, b, c, d, e, f, g))
        return random.choice(filtered if filtered else solutions)

    elif SAT_type in ('MAX2SAT', 'MAX2SAT0.2', 'MAX2SAT0.4', 'MAX2SAT0.6', 'MAX2SAT0.8', 'MAX2SAT1'):
        low, high = ratio_range_MAX2SAT[SAT_type]
        for a, b, c, d, e, f, g in solutions:
            total = a + b + c + d + e + f + g
            ratio = g / total
            if low <= ratio <= high:
                filtered.append((a, b, c, d, e, f, g))
        return random.choice(filtered if filtered else solutions)
    elif SAT_type == 1:
        return random.choice(solutions)
    elif SAT_type == 2:
        return random.choice(solutions)
    elif SAT_type == 3:
        return random.choice(solutions)


# 1.1 Generate neuron states in learn phase /Generate Random Logical Rule according to the logical rule
def generate_learn_neuron_state(Numliterals1, Numliterals1_2, Numliterals1_2_3, NewNumliterals1_2_3_4, Numliterals1_2_3_4_Red3, Numliterals1_2_3_4_Red3_Red2, TolTraNumliterals_MAX2SAT,
                                NumTotClause1_2_3, NumTotClause1_2_3_4, NeuronString, NeuronState, LogicRule):  # randomized neuron states
    for i in range(Numliterals1):
        for j in range(NeuronString):
            NeuronState[i][j] = random.choice([-1, 1])
    for i in range(Numliterals1, Numliterals1_2):
        for j in range(NeuronString):
            NeuronState[i][j] = random.choice([-1, 1])
    for i in range(Numliterals1_2, Numliterals1_2_3):
        for j in range(NeuronString):
            NeuronState[i][j] = random.choice([-1, 1])
    # 0 : A V B V C V D  <=>   A V B <=> P     # 1 : A V B V C V _D  <=>  (A V B) <=> P     # 2 : A V B V _C V D  <=>  (A V B) <=> P
    # 3 : A V B V _C V _D  <=>  (A V B) <=> P     # 4 : A V _B V C V D  <=>  (A V _B) <=> P     # 5 : A V _B V C V _D  <=>  (A V _B) <=> P
    # 6 : A V _B V _C V D <=>  (A V _B) <=> P     # 7 :A V _B V _C V _D  <=>  (A V _B) <=> P     # 8 : _A V B V C V D <=> (_A V B) <=> P
    # 9 : _A V B V C V _D  <=>  (_A V B) <=> P     # 10 : _A V B V _C V D  <=>  (_A V B) <=> P     # 11 : _A V B V _C V _D <=> (_A V B) <=> P
    # 12 : _A V _B V C V D  <=>  (_A V _B) <=> P     # 13 : _A V _B V C V _D  <=>  (_A V _B) <=> P     # 14 : _A V _B V _C V D  <=>  (_A V _B) <=> P
    # 15 : _A V _B V _C V _D  <=>  (_A V _B) <=> P
    valid_states3 = {
        0: (-1, -1), 1: (-1, -1), 2: (-1, -1), 3: (-1, -1), 4: (-1, 1), 5: (-1, 1),
        6: (-1, 1), 7: (-1, 1), 8: (1, -1), 9: (1, -1), 10: (1, -1), 11: (1, -1),
        12: (1, 1), 13: (1, 1), 14: (1, 1), 15: (1, 1), }
    for i in range(NumTotClause1_2_3, NumTotClause1_2_3_4):
        for j in range(NeuronString):
            state = LogicRule[i]
            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3][j] = random.choice([-1, 1])
            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 1][j] = random.choice([-1, 1])
            a = NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3][j]
            b = NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 1][j]
            if (a, b) != valid_states3.get(state):
                NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 2][j] = 1
                NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 3][j] = 1
            else:
                NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 2][j] = -1
                NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 3][j] = -1
            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 4][j] = random.choice([-1, 1])
            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 5][j] = random.choice([-1, 1])

            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 6][j] = \
                NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3][j]
            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 7][j] = \
                NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 3][j]

            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 8][j] = \
                NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 1][j]
            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 9][j] = \
                NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 3][j]

    for i in range(NewNumliterals1_2_3_4, Numliterals1_2_3_4_Red3):
        for j in range(NeuronString):
            NeuronState[i][j] = random.choice([-1, 1])
    for i in range(Numliterals1_2_3_4_Red3, Numliterals1_2_3_4_Red3_Red2):
        for j in range(NeuronString):
            NeuronState[i][j] = random.choice([-1, 1])
    for i in range(Numliterals1_2_3_4_Red3_Red2, TolTraNumliterals_MAX2SAT):
        for j in range(NeuronString):
            NeuronState[i][j] = random.choice([-1, 1])


# 1.2 Generate neuron states in retrieval phase
def generate_retrieval_neuron_state(TraNumliterals, NeuronString, NeuronState):  # randomized neuron states
    for i in range(TraNumliterals):
        for j in range(NeuronString):
            NeuronState[i][j] = random.choice([-1, 1])


# 1.2.2 Generate Random Logical Rule
def generate_logical_rule(NumClause1, NumTotClause1_2, NumTotClause1_2_3, NumTotClause1_2_3_4, NumTotClause1_2_3_4_Red3,
                          NumTotClause1_2_3_4_Red3_Red2,
                          NumTotClause_MAX2SAT, LogicRule):
    for i in range(NumClause1):
        LogicRule[i] = random.randint(0, 1)  # 0 : A     # 1: _A
    for i in range(NumClause1, NumTotClause1_2):
        LogicRule[i] = random.randint(0, 3)  # 0 : A V B    # 1 : A V _B    # 2 : _A V B    # 3 : _A V _B
    # 0 : A V B V C    # 1 : A V B V _C    # 2 : A V _B V C    # 3 : A V _B V _C
    # 4 : _A V B V C    # 5 : _A V  B V _C    # 6 : _A V _B V C    # 7 : _A V _B V _C
    for i in range(NumTotClause1_2, NumTotClause1_2_3):
        LogicRule[i] = random.randint(0, 7)
    for i in range(NumTotClause1_2_3, NumTotClause1_2_3_4):
        LogicRule[i] = random.randint(0, 15)
    # 0 : (A V B V C) ∧ (A V B V _C) = A V B    # 1 : (A V B V C) ∧ (A V _B V C) = A V C
    # 2 : (A V B V C) ∧ (_A V B V C) = B V C    # 3 : (_A V _B V _C) ∧ (_A V _B V C) = _A V _B
    # 4 : (_A V _B V _C) ∧ (_A V B V _C) = _A V _C    # 5 : (_A V _B V _C) ∧ (A V _B V _C) = _B V _C
    # 6 : (A V _B V _C) ∧ (A V _B V C) = A V _B    # 7 : (A V _B V _C) ∧ (A V B V _C) = A V _C
    # 8 : (_A V B V C) ∧ (_A V B V _C) = _A V B    # 9 : (A V B V _C) ∧ (_A V B V _C) = B V _C
    # 10 : (_A V B V C) ∧ (_A V _B V C) = _A V C    # 11 : (A V _B V C) ∧ (_A V _B V C) = _B V C
    for i in range(NumTotClause1_2_3_4, NumTotClause1_2_3_4_Red3):
        LogicRule[i] = random.randint(0, 11)
    # 0 : (A V B) ∧ (A V _B) = A    # 1 : (A V B) ∧ (_A V B) = B
    # 2 : (_A V _B) ∧ (_A V B) = _A    # 3 : (_A V _B) ∧ (A V _B) = _B
    for i in range(NumTotClause1_2_3_4_Red3, NumTotClause1_2_3_4_Red3_Red2):
        LogicRule[i] = random.randint(0, 3)
    # 0 : (_p V _q) ∧ (_p V q) ∧ (p V _q) ∧ (p V _r) ∧ (q V r)     # 1 : (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (q V _r) ∧ (p V r)
    # 2: (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_p V _r) ∧ (_q V r)      # 3 : (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_q V _r) ∧ (_p V r)
    for i in range(NumTotClause1_2_3_4_Red3_Red2, NumTotClause_MAX2SAT):
        LogicRule[i] = random.randint(0, 3)


# 1.3 Two types for ablation analysis
# 1.3.1 - Calculate Fitness of the solution according to cost function
def calculate_fitness(stringNum, NumClause1, NumTotClause1_2, NumTotClause1_2_3, NumTotClause1_2_3_4, NumTotClause1_2_3_4_Red3, NumTotClause1_2_3_4_Red3_Red2, NumTotClause_MAX2SAT,
                      Numliterals1_2, Numliterals1_2_3, NewNumliterals1_2_3_4, Numliterals1_2_3_4_Red3, Numliterals1_2_3_4_Red3_Red2, LogicRule, NeuronState):
    # calculate the maximum fitness value
    OriPoint = 0
    Red3Point = 0
    Red2Point = 0
    TseitinPoint = 0
    MAX2SATPoint = 0
    # calculate fitness for NumClauses1
    valid_states1 = {
        0: -1,
        1: 1,
    }
    for i in range(NumClause1):
        state = LogicRule[i]
        a = NeuronState[i * 1][stringNum]
        if a != valid_states1.get(state):
            OriPoint += 1
    # calculate fitness for NumClauses2
    valid_states2 = {
        0: (-1, -1),
        1: (-1, 1),
        2: (1, -1),
        3: (1, 1),
    }
    for i in range(NumClause1, NumTotClause1_2):
        state = LogicRule[i]
        a = NeuronState[(i - NumClause1) * 2 + NumClause1][stringNum]
        b = NeuronState[(i - NumClause1) * 2 + NumClause1 + 1][stringNum]
        if (a, b) != valid_states2.get(state):
            OriPoint += 1
    # calculate fitness for NumClauses3
    valid_states3 = {
        0: (-1, -1, -1),
        1: (-1, -1, 1),
        2: (-1, 1, -1),
        3: (-1, 1, 1),
        4: (1, -1, -1),
        5: (1, -1, 1),
        6: (1, 1, -1),
        7: (1, 1, 1),
    }
    for i in range(NumTotClause1_2, NumTotClause1_2_3):
        state = LogicRule[i]
        a, b, c = NeuronState[(i - NumTotClause1_2) * 3 + Numliterals1_2][stringNum], \
            NeuronState[(i - NumTotClause1_2) * 3 + Numliterals1_2 + 1][stringNum], \
            NeuronState[(i - NumTotClause1_2) * 3 + Numliterals1_2 + 2][stringNum]
        if (a, b, c) != valid_states3.get(state):
            OriPoint += 1
    # calculate fitness for NumClauses4
    # 0 : A V B V C V D  <=>  (A V B) <=> P     # 1 : A V B V C V _D  <=>  (A V B) <=> P    # 2 : A V B V _C V D  <=>  (A V B) <=> P    # 3 : A V B V _C V _D  <=>  (A V B) <=> P
    # 4 : A V _B V C V D  <=>  (A V _B) <=> P    # 5 : A V _B V C V _D  <=>  (A V _B) <=> P    # 6 : A V _B V _C V D <=>  (A V _B) <=> P    # 7 :A V _B V _C V _D  <=>  (A V _B) <=> P
    # 8 : _A V B V C V D <=>  (_A V B) <=> P     # 9 : _A V B V C V _D  <=>  (_A V B) <=> P    # 10 : _A V B V _C V D  <=>  (_A V B) <=> P    # 11 : _A V B V _C V _D <=> (_A V B) <=> P
    # 12 : _A V _B V C V D  <=>  (_A V _B) <=> P    # 13 : _A V _B V C V _D  <=>  (_A V _B) <=> P    # 14 : _A V _B V _C V D  <=>  (_A V _B) <=> P    # 15 : _A V _B V _C V _D  <=>  (_A V _B) <=> P
    valid_states4 = {
        0: (-1, -1, -1, -1),
        1: (-1, -1, -1, 1),
        2: (-1, -1, 1, -1),
        3: (-1, -1, 1, 1),
        4: (-1, 1, -1, -1),
        5: (-1, 1, -1, 1),
        6: (-1, 1, 1, -1),
        7: (-1, 1, 1, 1),
        8: (1, -1, -1, -1),
        9: (1, -1, -1, 1),
        10: (1, -1, 1, -1),
        11: (1, -1, 1, 1),
        12: (1, 1, -1, -1),
        13: (1, 1, -1, 1),
        14: (1, 1, 1, -1),
        15: (1, 1, 1, 1),
    }
    for i in range(NumTotClause1_2_3, NumTotClause1_2_3_4):
        state = LogicRule[i]
        a, b, c, d = NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3][stringNum], \
            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 1][stringNum], \
            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 4][stringNum], \
            NeuronState[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 5][stringNum]
        TseitinPoint += 3
        if (a, b, c, d) != valid_states4.get(state):
            OriPoint += 1
    # calculate fitness for NumRedClause3
    valid_states_red3 = {
        0: (lambda a, b, c: a == -1 and b == -1),  # 0 : (A V B V C) ∧ (A V B V _C) = A V B
        1: (lambda a, b, c: a == -1 and c == -1),  # 1 : (A V B V C) ∧ (A V _B V C) = A V C
        2: (lambda a, b, c: b == -1 and c == -1),  # 2 : (A V B V C) ∧ (_A V B V C) = B V C
        3: (lambda a, b, c: a == 1 and b == 1),  # 3 : (_A V _B V _C) ∧ (_A V _B V C) = _A V _B
        4: (lambda a, b, c: a == 1 and c == 1),  # 4 : (_A V _B V _C) ∧ (_A V B V _C) = _A V _C
        5: (lambda a, b, c: b == 1 and c == 1),  # 5 : (_A V _B V _C) ∧ (A V _B V _C) = _B V _C
        6: (lambda a, b, c: a == -1 and b == 1),  # 6 : (A V _B V _C) ∧ (A V _B V C) = A V _B
        7: (lambda a, b, c: a == -1 and c == 1),  # 7 : (A V _B V _C) ∧ (A V B V _C) = A V _C
        8: (lambda a, b, c: a == 1 and b == -1),  # 8 : (_A V B V  C) ∧ (_A V B V _C) = _A V B
        9: (lambda a, b, c: b == -1 and c == 1),  # 9 : (A V B V _C) ∧ (_A V B V _C) = B V _C
        10: (lambda a, b, c: a == 1 and c == -1),  # 10 : (_A V B V C) ∧ (_A V _B V C) = _A V C
        11: (lambda a, b, c: b == 1 and c == -1),  # 11 : (A V _B V C) ∧ (_A V _B V C) = _B V C
    }
    for i in range(NumTotClause1_2_3_4, NumTotClause1_2_3_4_Red3):
        state = LogicRule[i]
        a = NeuronState[(i - NumTotClause1_2_3_4) * 3 + NewNumliterals1_2_3_4][stringNum]
        b = NeuronState[(i - NumTotClause1_2_3_4) * 3 + 1 + NewNumliterals1_2_3_4][stringNum]
        c = NeuronState[(i - NumTotClause1_2_3_4) * 3 + 2 + NewNumliterals1_2_3_4][stringNum]
        condition = valid_states_red3[state]
        if not condition(a, b, c):
            OriPoint += 1
            Red3Point += 1
    # calculate fitness for NumRedClause2
    # 0 : (A V B) ∧ (A V _B) = A    # 1 : (A V B) ∧ (_A V B) = B
    # 2 : (_A V _B) ∧ (_A V B) = _A    # 3 : (_A V _B) ∧ (A V _B) = _B
    valid_states_red2 = {
        0: (lambda a, b: a == -1),  # 0 : (A V B) ∧ (A V _B) = A
        1: (lambda a, b: b == -1),  # 1 : (A V B) ∧ (_A V B) = B
        2: (lambda a, b: a == 1),  # 2 : (_A V _B) ∧ (_A V B) = _A
        3: (lambda a, b: b == 1),  # 3 : (_A V _B) ∧ (A V _B) = _B
    }
    for i in range(NumTotClause1_2_3_4_Red3, NumTotClause1_2_3_4_Red3_Red2):
        state = LogicRule[i]
        a = NeuronState[(i - NumTotClause1_2_3_4_Red3) * 2 + Numliterals1_2_3_4_Red3][stringNum]
        b = NeuronState[(i - NumTotClause1_2_3_4_Red3) * 2 + 1 + Numliterals1_2_3_4_Red3][stringNum]
        condition = valid_states_red2[state]
        if not condition(a, b):
            OriPoint += 1
            Red2Point += 1
    # calculate fitness for NumClauseMAX2SAT
    # 0 : (_p V _q) ∧ (_p V q) ∧ (p V _q) ∧ (p V _r) ∧ (q V r)     # 1 : (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (q V _r) ∧ (p V r)
    # 2: (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_p V _r) ∧ (_q V r)      # 3 : (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_q V _r) ∧ (_p V r)
    valid_states_MAX2SAT = {
        0: {(1, -1, -1), (-1, 1, 1)},
        1: {(1, -1, 1), (-1, 1, -1)},
        2: {(1, -1, 1), (-1, 1, -1)},
        3: {(1, -1, -1), (-1, 1, 1)},
    }
    for i in range(NumTotClause1_2_3_4_Red3_Red2, NumTotClause_MAX2SAT):
        state = LogicRule[i]
        a = NeuronState[(i - NumTotClause1_2_3_4_Red3_Red2) * 3 + Numliterals1_2_3_4_Red3_Red2][stringNum]
        b = NeuronState[(i - NumTotClause1_2_3_4_Red3_Red2) * 3 + 1 + Numliterals1_2_3_4_Red3_Red2][stringNum]
        c = NeuronState[(i - NumTotClause1_2_3_4_Red3_Red2) * 3 + 2 + Numliterals1_2_3_4_Red3_Red2][stringNum]
        allowed = valid_states_MAX2SAT[state]
        MAX2SATPoint += 3
        if (a, b, c) not in allowed:
            OriPoint += 1

    NumTotalNumComponent = NumTotClause_MAX2SAT
    # Initial Formulation
    InitialTotalNumClauseNoSAT = NumTotClause1_2_3_4 + (NumTotClause1_2_3_4_Red3_Red2 - NumTotClause1_2_3_4) * 2 + (NumTotClause_MAX2SAT - NumTotClause1_2_3_4_Red3_Red2) * 5
    InitialTotalNumClauseSAT = InitialTotalNumClauseNoSAT - (NumTotClause_MAX2SAT - NumTotClause1_2_3_4_Red3_Red2)
    # (1) Redundancy elimination
    RedTotalNumClauseNoSAT = NumTotClause1_2_3_4_Red3_Red2 + (NumTotClause_MAX2SAT - NumTotClause1_2_3_4_Red3_Red2) * 5
    RedTotalNumClauseSAT = RedTotalNumClauseNoSAT - (NumTotClause_MAX2SAT - NumTotClause1_2_3_4_Red3_Red2)
    # (2) Tseitin transformation
    TseitinTotalNumClauseNoSAT = NumTotClause1_2_3 + (NumTotClause1_2_3_4 - NumTotClause1_2_3) * 4 + (NumTotClause1_2_3_4_Red3_Red2 - NumTotClause1_2_3_4) * 2 + (NumTotClause_MAX2SAT - NumTotClause1_2_3_4_Red3_Red2) * 5
    TseitinTotalNumClauseSAT = TseitinTotalNumClauseNoSAT - (NumTotClause_MAX2SAT - NumTotClause1_2_3_4_Red3_Red2)
    # After (1) and (2)
    AfterTotalNumClauseNoSAT = NumTotClause1_2_3 + (NumTotClause1_2_3_4 - NumTotClause1_2_3) * 4 + (NumTotClause1_2_3_4_Red3_Red2 - NumTotClause1_2_3_4) + (NumTotClause_MAX2SAT - NumTotClause1_2_3_4_Red3_Red2) * 5
    AfterTotalNumClauseSAT = AfterTotalNumClauseNoSAT - (NumTotClause_MAX2SAT - NumTotClause1_2_3_4_Red3_Red2)

    # print(OriPoint + Red3Point + Red2Point + MAX2SATPoint, InitialTotalNumClauseNoSAT)

    return (OriPoint, OriPoint / NumTotalNumComponent,
            (OriPoint + Red3Point + Red2Point + MAX2SATPoint) / InitialTotalNumClauseNoSAT, (OriPoint + Red3Point + Red2Point + MAX2SATPoint) / InitialTotalNumClauseSAT,
            (OriPoint + MAX2SATPoint) / RedTotalNumClauseNoSAT, (OriPoint + MAX2SATPoint) / RedTotalNumClauseSAT,
            (OriPoint + TseitinPoint + Red3Point + Red2Point + MAX2SATPoint) / TseitinTotalNumClauseNoSAT, (OriPoint + TseitinPoint + Red3Point + Red2Point + MAX2SATPoint) / TseitinTotalNumClauseSAT,
            (OriPoint + TseitinPoint + MAX2SATPoint) / AfterTotalNumClauseNoSAT, (OriPoint + TseitinPoint + MAX2SATPoint) / AfterTotalNumClauseSAT)


#  1.4 - Check the highest Fitness of the String
def print_point_max(Numliterals, FitnessNum, Result, NeuronState):
    max_point = max(FitnessNum)
    index_of_max = np.argmax(FitnessNum)
    for j in range(Numliterals):
        Result[j] = NeuronState[j][index_of_max]
    return max_point, index_of_max


# PART 2 Learning Algorithm
# 2.1 - Exhaustive Search


# PART 3 Similarity Analysis
def check_similarity(LogicRule, NeuronState, Numliterals1, Numliterals1_2, Numliterals1_2_3, Numliterals1_2_3_4, Numliterals1_2_3_4_Red3, Numliterals1_2_3_4_Red3_Red2, Numliterals_MAX2SAT,
                     NumClause1, NumClause1_2, NumClause1_2_3, NumClause1_2_3_4, NumClause1_2_3_4_Red3, NumClause1_2_3_4_Red3_Red2):
    l = 0
    m = 0
    n = 0
    o = 0
    # deal with redundant variables
    Red3l = 0
    Red3m = 0
    Red3n = 0
    Red3o = 0

    Red2l = 0
    Red2m = 0
    Red2n = 0
    Red2o = 0
    # deal with Tseitin
    Tsel = 0
    Tsem = 0
    Tsen = 0
    Tseo = 0
    # 1-SAT
    for i in range(Numliterals1):
        clause_type = LogicRule[i]
        if clause_type == 0:  # A
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
        elif clause_type == 1:  # _A
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
    # 2-SAT
    for i in range(Numliterals1, Numliterals1_2, 2):
        idx = (i - Numliterals1) // 2 + NumClause1
        clause_type = LogicRule[idx]
        if clause_type == 0:  # 1 1
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
        elif clause_type == 1:  # 1 -1
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
        elif clause_type == 2:  # -1 1
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
        else:  # -1 -1
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
    # 3-SAT
    for i in range(Numliterals1_2, Numliterals1_2_3, 3):
        idx = (i - Numliterals1_2) // 3 + NumClause1_2
        clause_type = LogicRule[idx]
        # 0 : A V B V C    # 1 : A V B V _C    # 2 : A V _B V C    # 3 : A V _B V _C
        # 4 : _A V B V C    # 5 : _A V  B V _C    # 6 : _A V _B V C    # 7 : _A V _B V _C
        if clause_type == 0:  # A V B V C
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)

        elif clause_type == 1:  # A V B V _C
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)

        elif clause_type == 2:  # A V _B V C
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)

        elif clause_type == 3:  # A V _B V _C
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)

        elif clause_type == 4:  # _A V B V C
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)

        elif clause_type == 5:  # _A V B V _C
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)

        elif clause_type == 6:  # _A V _B V C
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)

        elif clause_type == 7:  # _A V _B V _C
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)
    # 4-SAT
    # 0 : A V B V C V D      # 1 : A V B V C V _D    # 2 : A V B V _C V D     # 3 : A V B V _C V _D
    # 4 : A V _B V C V D    # 5 : A V _B V C V _D     # 6 : A V _B V _C V D    # 7 :A V _B V _C V _D
    for i in range(Numliterals1_2_3, Numliterals1_2_3_4, 10):
        idx = (i - Numliterals1_2_3) // 10 + NumClause1_2_3
        clause_type = LogicRule[idx]
        if clause_type == 0:  # A V B V C V D
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            l += int(NeuronState[i + 4] == 1)
            m += int(NeuronState[i + 4] != 1)
            l += int(NeuronState[i + 5] == 1)
            m += int(NeuronState[i + 5] != 1)

            Tsen += int(NeuronState[i + 6] == 1)
            Tseo += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsen += int(NeuronState[i + 8] == 1)
            Tseo += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)

        elif clause_type == 1:  # A V B V C V _D
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            l += int(NeuronState[i + 4] == 1)
            m += int(NeuronState[i + 4] != 1)
            n += int(NeuronState[i + 5] == 1)
            o += int(NeuronState[i + 5] != 1)

            Tsen += int(NeuronState[i + 6] == 1)
            Tseo += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsen += int(NeuronState[i + 8] == 1)
            Tseo += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)

        elif clause_type == 2:  # A V B V _C V D
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            n += int(NeuronState[i + 4] == 1)
            o += int(NeuronState[i + 4] != 1)
            l += int(NeuronState[i + 5] == 1)
            m += int(NeuronState[i + 5] != 1)

            Tsen += int(NeuronState[i + 6] == 1)
            Tseo += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsen += int(NeuronState[i + 8] == 1)
            Tseo += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)

        elif clause_type == 3:  # A V B V _C V _D
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            n += int(NeuronState[i + 4] == 1)
            o += int(NeuronState[i + 4] != 1)
            n += int(NeuronState[i + 5] == 1)
            o += int(NeuronState[i + 5] != 1)

            Tsen += int(NeuronState[i + 6] == 1)
            Tseo += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsen += int(NeuronState[i + 8] == 1)
            Tseo += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 4:  # A V _B V C V D
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            l += int(NeuronState[i + 4] == 1)
            m += int(NeuronState[i + 4] != 1)
            l += int(NeuronState[i + 5] == 1)
            m += int(NeuronState[i + 5] != 1)

            Tsen += int(NeuronState[i + 6] == 1)
            Tseo += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsel += int(NeuronState[i + 8] == 1)
            Tsem += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 5:  # A V _B V C V _D
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            l += int(NeuronState[i + 4] == 1)
            m += int(NeuronState[i + 4] != 1)
            n += int(NeuronState[i + 5] == 1)
            o += int(NeuronState[i + 5] != 1)

            Tsen += int(NeuronState[i + 6] == 1)
            Tseo += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsel += int(NeuronState[i + 8] == 1)
            Tsem += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 6:  # A V _B V _C V D
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            n += int(NeuronState[i + 4] == 1)
            o += int(NeuronState[i + 4] != 1)
            l += int(NeuronState[i + 5] == 1)
            m += int(NeuronState[i + 5] != 1)

            Tsen += int(NeuronState[i + 6] == 1)
            Tseo += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsel += int(NeuronState[i + 8] == 1)
            Tsem += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 7:  # A V _B V _C V _D
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            n += int(NeuronState[i + 4] == 1)
            o += int(NeuronState[i + 4] != 1)
            n += int(NeuronState[i + 5] == 1)
            o += int(NeuronState[i + 5] != 1)

            Tsen += int(NeuronState[i + 6] == 1)
            Tseo += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsel += int(NeuronState[i + 8] == 1)
            Tsem += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        # 8 : _A V B V C V D
        # 9 : _A V B V C V _D    # 10 : _A V B V _C V D    # 11 : _A V B V _C V _D
        # 12 : _A V _B V C V D    # 13 : _A V _B V C V _D    # 14 : _A V _B V _C V D
        # 15 : _A V _B V _C V _D
        elif clause_type == 8:  # _A V B V C V D
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            l += int(NeuronState[i + 4] == 1)
            m += int(NeuronState[i + 4] != 1)
            l += int(NeuronState[i + 5] == 1)
            m += int(NeuronState[i + 5] != 1)

            Tsel += int(NeuronState[i + 6] == 1)
            Tsem += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsen += int(NeuronState[i + 8] == 1)
            Tseo += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 9:  # _A V B V C V _D
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            l += int(NeuronState[i + 4] == 1)
            m += int(NeuronState[i + 4] != 1)
            n += int(NeuronState[i + 5] == 1)
            o += int(NeuronState[i + 5] != 1)

            Tsel += int(NeuronState[i + 6] == 1)
            Tsem += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsen += int(NeuronState[i + 8] == 1)
            Tseo += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 10:  # _A V B V _C V D
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            n += int(NeuronState[i + 4] == 1)
            o += int(NeuronState[i + 4] != 1)
            l += int(NeuronState[i + 5] == 1)
            m += int(NeuronState[i + 5] != 1)

            Tsel += int(NeuronState[i + 6] == 1)
            Tsem += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsen += int(NeuronState[i + 8] == 1)
            Tseo += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 11:  # _A V B V _C V _D
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            n += int(NeuronState[i + 4] == 1)
            o += int(NeuronState[i + 4] != 1)
            n += int(NeuronState[i + 5] == 1)
            o += int(NeuronState[i + 5] != 1)

            Tsel += int(NeuronState[i + 6] == 1)
            Tsem += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsen += int(NeuronState[i + 8] == 1)
            Tseo += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 12:  # _A V _B V C V D
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            l += int(NeuronState[i + 4] == 1)
            m += int(NeuronState[i + 4] != 1)
            l += int(NeuronState[i + 5] == 1)
            m += int(NeuronState[i + 5] != 1)

            Tsel += int(NeuronState[i + 6] == 1)
            Tsem += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsel += int(NeuronState[i + 8] == 1)
            Tsem += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 13:  # _A V _B V C V _D
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            l += int(NeuronState[i + 4] == 1)
            m += int(NeuronState[i + 4] != 1)
            n += int(NeuronState[i + 5] == 1)
            o += int(NeuronState[i + 5] != 1)

            Tsel += int(NeuronState[i + 6] == 1)
            Tsem += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsel += int(NeuronState[i + 8] == 1)
            Tsem += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 14:  # _A V _B V _C V D
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            n += int(NeuronState[i + 4] == 1)
            o += int(NeuronState[i + 4] != 1)
            l += int(NeuronState[i + 5] == 1)
            m += int(NeuronState[i + 5] != 1)

            Tsel += int(NeuronState[i + 6] == 1)
            Tsem += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsel += int(NeuronState[i + 8] == 1)
            Tsem += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
        elif clause_type == 15:  # _A V _B V _C V _D
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            Tsen += int(NeuronState[i + 2] == 1)
            Tseo += int(NeuronState[i + 2] != 1)
            Tsel += int(NeuronState[i + 3] == 1)
            Tsem += int(NeuronState[i + 3] != 1)

            n += int(NeuronState[i + 4] == 1)
            o += int(NeuronState[i + 5] != 1)
            n += int(NeuronState[i + 4] == 1)
            o += int(NeuronState[i + 5] != 1)

            Tsel += int(NeuronState[i + 6] == 1)
            Tsem += int(NeuronState[i + 6] != 1)
            Tsel += int(NeuronState[i + 7] == 1)
            Tsem += int(NeuronState[i + 7] != 1)

            Tsel += int(NeuronState[i + 8] == 1)
            Tsem += int(NeuronState[i + 8] != 1)
            Tsel += int(NeuronState[i + 9] == 1)
            Tsem += int(NeuronState[i + 9] != 1)
    # Red3-SAT
    # 0 : (A V B V C) ∧ (A V B V _C) = A V B    # 1 : (A V B V C) ∧ (A V _B V C) = A V C     # 2 : (A V B V C) ∧ (_A V B V C) = B V C
    # 3 : (_A V _B V _C) ∧ (_A V _B V C) = _A V _B     # 4 : (_A V _B V _C) ∧ (_A V B V _C) = _A V _C    # 5 : (_A V _B V _C) ∧ (A V _B V _C) = _B V _C
    for i in range(Numliterals1_2_3_4, Numliterals1_2_3_4_Red3, 3):
        idx = (i - Numliterals1_2_3_4) // 3 + NumClause1_2_3_4
        clause_type = LogicRule[idx]
        if clause_type == 0:  # (A V B V C) ∧ (A V B V _C) = A V B
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            Red3l += int(NeuronState[i + 2] == 1)
            Red3m += int(NeuronState[i + 2] != 1)

            Red3l += int(NeuronState[i] == 1)
            Red3m += int(NeuronState[i] != 1)
            Red3l += int(NeuronState[i + 1] == 1)
            Red3m += int(NeuronState[i + 1] != 1)
            Red3n += int(NeuronState[i + 2] == 1)
            Red3o += int(NeuronState[i + 2] != 1)

        elif clause_type == 1:  # (A V B V C) ∧ (A V _B V C) = A V C
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)
            #
            Red3l += int(NeuronState[i + 1] == 1)
            Red3m += int(NeuronState[i + 1] != 1)

            Red3l += int(NeuronState[i] == 1)
            Red3m += int(NeuronState[i] != 1)
            Red3n += int(NeuronState[i + 1] == 1)
            Red3o += int(NeuronState[i + 1] != 1)
            Red3l += int(NeuronState[i + 2] == 1)
            Red3m += int(NeuronState[i + 2] != 1)

        elif clause_type == 2:  # (A V B V C) ∧ (_A V B V C) = B V C
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)
            #
            Red3l += int(NeuronState[i] == 1)
            Red3m += int(NeuronState[i] != 1)

            Red3n += int(NeuronState[i] == 1)
            Red3o += int(NeuronState[i] != 1)
            Red3l += int(NeuronState[i + 1] == 1)
            Red3m += int(NeuronState[i + 1] != 1)
            Red3l += int(NeuronState[i + 2] == 1)
            Red3m += int(NeuronState[i + 2] != 1)

        elif clause_type == 3:  # (_A V _B V _C) ∧ (_A V _B V C) = _A V _B
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            #
            Red3n += int(NeuronState[i + 2] == 1)
            Red3o += int(NeuronState[i + 2] != 1)

            Red3n += int(NeuronState[i] == 1)
            Red3o += int(NeuronState[i] != 1)
            Red3n += int(NeuronState[i + 1] == 1)
            Red3o += int(NeuronState[i + 1] != 1)
            Red3l += int(NeuronState[i + 2] == 1)
            Red3m += int(NeuronState[i + 2] != 1)

        elif clause_type == 4:  # (_A V _B V _C) ∧ (_A V B V _C) = _A V _C
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)
            #
            Red3n += int(NeuronState[i + 1] == 1)
            Red3o += int(NeuronState[i + 1] != 1)

            Red3n += int(NeuronState[i] == 1)
            Red3o += int(NeuronState[i] != 1)
            Red3l += int(NeuronState[i + 1] == 1)
            Red3m += int(NeuronState[i + 1] != 1)
            Red3n += int(NeuronState[i + 2] == 1)
            Red3o += int(NeuronState[i + 2] != 1)

        elif clause_type == 5:  # (_A V _B V _C) ∧ (A V _B V _C) = _B V _C
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)
            #
            Red3n += int(NeuronState[i] == 1)
            Red3o += int(NeuronState[i] != 1)

            Red3l += int(NeuronState[i] == 1)
            Red3m += int(NeuronState[i] != 1)
            Red3n += int(NeuronState[i + 1] == 1)
            Red3o += int(NeuronState[i + 1] != 1)
            Red3n += int(NeuronState[i + 2] == 1)
            Red3o += int(NeuronState[i + 2] != 1)
        # 6 : (A V _B V _C) ∧ (A V _B V C) = A V _B    # 7 : (A V _B V _C) ∧ (A V B V _C) = A V _C        # 8 : (_A V B V C) ∧ (_A V B V _C) = _A V B
        # 9 : (A V B V _C) ∧ (_A V B V _C) = B V _C        # 10 : (_A V B V C) ∧ (_A V _B V C) = _A V C    # 11 : (A V _B V C) ∧ (_A V _B V C) = _B V C
        elif clause_type == 6:  # (A V _B V _C) ∧ (A V _B V C) = A V _B
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            #
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)

            Red3l += int(NeuronState[i] == 1)
            Red3m += int(NeuronState[i] != 1)
            Red3n += int(NeuronState[i + 1] == 1)
            Red3o += int(NeuronState[i + 1] != 1)
            Red3l += int(NeuronState[i + 2] == 1)
            Red3m += int(NeuronState[i + 2] != 1)

        elif clause_type == 7:  # (A V _B V _C) ∧ (A V B V _C) = A V _C
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)
            #
            Red3n += int(NeuronState[i + 1] == 1)
            Red3o += int(NeuronState[i + 1] != 1)

            Red3l += int(NeuronState[i] == 1)
            Red3m += int(NeuronState[i] != 1)
            Red3l += int(NeuronState[i + 1] == 1)
            Red3m += int(NeuronState[i + 1] != 1)
            Red3n += int(NeuronState[i + 2] == 1)
            Red3o += int(NeuronState[i + 2] != 1)

        elif clause_type == 8:  # (_A V B V C) ∧ (_A V B V _C) = _A V B
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
            #
            Red3l += int(NeuronState[i + 2] == 1)
            Red3m += int(NeuronState[i + 2] != 1)

            Red3n += int(NeuronState[i] == 1)
            Red3o += int(NeuronState[i] != 1)
            Red3l += int(NeuronState[i + 1] == 1)
            Red3m += int(NeuronState[i + 1] != 1)
            Red3n += int(NeuronState[i + 2] == 1)
            Red3o += int(NeuronState[i + 2] != 1)

        elif clause_type == 9:  # (A V B V _C) ∧ (_A V B V _C) = B V _C
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)
            #
            Red3l += int(NeuronState[i] == 1)
            Red3m += int(NeuronState[i] != 1)

            Red3n += int(NeuronState[i] == 1)
            Red3o += int(NeuronState[i] != 1)
            Red3l += int(NeuronState[i + 1] == 1)
            Red3m += int(NeuronState[i + 1] != 1)
            Red3n += int(NeuronState[i + 2] == 1)
            Red3o += int(NeuronState[i + 2] != 1)

        elif clause_type == 10:  # (_A V B V C) ∧ (_A V _B V C) = _A V C
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)
            #
            Red3l += int(NeuronState[i + 1] == 1)
            Red3m += int(NeuronState[i + 1] != 1)

            Red3n += int(NeuronState[i] == 1)
            Red3o += int(NeuronState[i] != 1)
            Red3n += int(NeuronState[i + 1] == 1)
            Red3o += int(NeuronState[i + 1] != 1)
            Red3l += int(NeuronState[i + 2] == 1)
            Red3m += int(NeuronState[i + 2] != 1)

        elif clause_type == 11:  # (A V _B V C) ∧ (_A V _B V C) = _B V C
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)
            #
            Red3l += int(NeuronState[i] == 1)
            Red3m += int(NeuronState[i] != 1)

            Red3n += int(NeuronState[i] == 1)
            Red3o += int(NeuronState[i] != 1)
            Red3n += int(NeuronState[i + 1] == 1)
            Red3o += int(NeuronState[i + 1] != 1)
            Red3l += int(NeuronState[i + 2] == 1)
            Red3m += int(NeuronState[i + 2] != 1)
    # Red2-SAT
    # 0 : (A V B) ∧ (A V _B) = A    # 1 : (A V B) ∧ (_A V B) = B    # 2 : (_A V _B) ∧ (_A V B) = _A    # 3 : (_A V _B) ∧ (A V _B) = _B
    for i in range(Numliterals1_2_3_4_Red3, Numliterals1_2_3_4_Red3_Red2, 2):
        idx = (i - Numliterals1_2_3_4_Red3) // 2 + NumClause1_2_3_4_Red3
        clause_type = LogicRule[idx]
        if clause_type == 0:  # 0 : (A V B) ∧ (A V _B) = A
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)

            Red2l += int(NeuronState[i + 1] == 1)
            Red2m += int(NeuronState[i + 1] != 1)

            Red2l += int(NeuronState[i] == 1)
            Red2m += int(NeuronState[i] != 1)
            Red2n += int(NeuronState[i + 1] == 1)
            Red2o += int(NeuronState[i + 1] != 1)

        elif clause_type == 1:  # 1 : (A V B) ∧ (_A V B) = B
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
            #
            Red2l += int(NeuronState[i] == 1)
            Red2m += int(NeuronState[i] != 1)

            Red2n += int(NeuronState[i] == 1)
            Red2o += int(NeuronState[i] != 1)
            Red2l += int(NeuronState[i + 1] == 1)
            Red2m += int(NeuronState[i + 1] != 1)
        elif clause_type == 2:  # 2 : (_A V _B) ∧ (_A V B) = _A
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            #
            Red2n += int(NeuronState[i + 1] == 1)
            Red2o += int(NeuronState[i + 1] != 1)

            Red2n += int(NeuronState[i] == 1)
            Red2o += int(NeuronState[i] != 1)
            Red2l += int(NeuronState[i + 1] == 1)
            Red2m += int(NeuronState[i + 1] != 1)
        else:  # 3 : (_A V _B) ∧ (A V _B) = _B
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            #
            Red2n += int(NeuronState[i] == 1)
            Red2o += int(NeuronState[i] != 1)

            Red2l += int(NeuronState[i] == 1)
            Red2m += int(NeuronState[i] != 1)
            Red2n += int(NeuronState[i + 1] == 1)
            Red2o += int(NeuronState[i + 1] != 1)
    # 0 : (_p V _q) ∧ (_p V q) ∧ (p V _q) ∧ (p V _r) ∧ (q V r)     # 1 : (_p V _q) ∧ (_p V q) ∧ (p V _q) ∧ (q V _r) ∧ (p V r)
    # 2: (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_p V _r) ∧ (_q V r)      # 3 : (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_q V _r) ∧ (_p V r)
    for i in range(Numliterals1_2_3_4_Red3_Red2, Numliterals_MAX2SAT, 3):
        idx = (i - Numliterals1_2_3_4_Red3_Red2) // 3 + NumClause1_2_3_4_Red3_Red2
        clause_type = LogicRule[idx]
        if clause_type == 0:  # 0 : (_p V _q) ∧ (_p V q) ∧ (p V _q) ∧ (p V _r) ∧ (q V r)
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)

            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)

        elif clause_type == 1:  # 1 : (_p V _q) ∧ (_p V q) ∧ (p V _q) ∧ (q V _r) ∧ (p V r)
            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)

            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)

        elif clause_type == 2:  # 2: (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_p V _r) ∧ (_q V r)
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)

            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)
        else:  # 3 : (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_q V _r) ∧ (_p V r)
            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 1] == 1)
            m += int(NeuronState[i + 1] != 1)

            l += int(NeuronState[i] == 1)
            m += int(NeuronState[i] != 1)
            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)

            n += int(NeuronState[i + 1] == 1)
            o += int(NeuronState[i + 1] != 1)
            n += int(NeuronState[i + 2] == 1)
            o += int(NeuronState[i + 2] != 1)

            n += int(NeuronState[i] == 1)
            o += int(NeuronState[i] != 1)
            l += int(NeuronState[i + 2] == 1)
            m += int(NeuronState[i + 2] != 1)

    InitialNum = l + m + n + Red3l + Red3m + Red3n + Red2l + Red2m + Red2n
    RedNum = l + m + n
    TraNum = l + m + n + Red3l + Red3m + Red3n + Red2l + Red2m + Red2n + Tsel + Tsem + Tsen
    AfterNum = l + m + n + Tsel + Tsem + Tsen
    if RedNum != 0:
        return ((l + Red3l + Red2l) / InitialNum, m + n + Red3m + Red3n + Red2m + Red2n,
                l / RedNum, m + n,
                (l + Red3l + Red2l + Tsel) / TraNum, m + n + Red3m + Red3n + Red2m + Red2n + Tsem + Tsen,
                (l + Tsel) / AfterNum, m + n + Tsem + Tsen)
    else:
        if AfterNum != 0 and InitialNum != 0:
            return (l + Red3l + Red2l) / InitialNum, m + n + Red3m + Red3n + Red2m + Red2n, 0, 0, (l + Red3l + Red2l + Tsel) / TraNum, m + n + Red3m + Red3n + Red2m + Red2n + Tsem + Tsen, (l + Tsel) / AfterNum, m + n + Tsem + Tsen
        elif AfterNum != 0 and InitialNum == 0:
            return 0, 0, 0, 0, (l + Red3l + Red2l + Tsel) / TraNum, m + n + Red3m + Red3n + Red2m + Red2n + Tsem + Tsen, (l + Tsel) / AfterNum, m + n + Tsem + Tsen
        elif AfterNum == 0 and InitialNum != 0:
            return (l + Red3l + Red2l) / InitialNum, m + n + Red3m + Red3n + Red2m + Red2n, 1, 0, (l + Red3l + Red2l + Tsel) / TraNum, m + n + Red3m + Red3n + Red2m + Red2n + Tsem + Tsen, 1, 0
        else:
            return 0, 0, 0, 0, 0, 0, 0, 0


#  PART 4 Hopfield Neural Networkdef check_similarity(LogicRule, NeuronState, Numliterals1, Numliterals1_2, Numliterals1_2_3, Numliterals1_2_3_4, Numliterals1_2_3_4_Red3, Numliterals1_2_3_4_Red3_Red2, Numliterals_MAX2SAT,
#                      NumClause1, NumClause1_2, NumClause1_2_3, NumClause1_2_3_4, NumClause1_2_3_4_Red3, NumClause1_2_3_4_Red3_Red2)
#  4.1 DHNN Function Body
# 4.1.1 Define Weights
def DHNN(weight, max_fitness, Result, NumClause1, NumTotClause1_2, NumTotClause1_2_3, NumTotClause1_2_3_4,
         NumTotClause1_2_3_4_Red3, NumTotClause1_2_3_4_Red3_Red2, NumTotClause_MAX2SAT, Numliterals1, Numliterals1_2,
         Numliterals1_2_3, NewNumliterals1_2_3_4, Numliterals1_2_3_4_Red3, Numliterals1_2_3_4_Red3_Red2, LogicRule):
    if max_fitness == NumTotClause_MAX2SAT:
        for i in range(NumClause1):
            weight.append(weight_1SAT[LogicRule[i]])
        for i in range(NumClause1, NumTotClause1_2):
            weight.append(weight_2SAT[LogicRule[i]])
        for i in range(NumTotClause1_2, NumTotClause1_2_3):
            weight.append(weight_3SAT[LogicRule[i]])
        for i in range(NumTotClause1_2_3, NumTotClause1_2_3_4):
            weight.append(weight_4SAT[LogicRule[i]])
        for i in range(NumTotClause1_2_3_4, NumTotClause1_2_3_4_Red3):
            weight.append(weight_3RedSAT[LogicRule[i]])
        for i in range(NumTotClause1_2_3_4_Red3, NumTotClause1_2_3_4_Red3_Red2):
            weight.append(weight_2RedSAT[LogicRule[i]])
        for i in range(NumTotClause1_2_3_4_Red3_Red2, NumTotClause_MAX2SAT):
            weight.append(weight_MAX2SAT[LogicRule[i]])
    else:
        # Randomly generate weights for 1SAT
        valid_states1 = {
            0: -1,
            1: 1,
        }
        for i in range(NumClause1):
            state = LogicRule[i]
            a = Result[i]
            if a != valid_states1.get(state):
                weight.append(weight_1SAT[state])
            else:
                W_1SAT = [random.uniform(-1, 1)]
                weight.append(W_1SAT)

        # Randomly generate weights for 2SAT
        valid_states2 = {
            0: (-1, -1),
            1: (-1, 1),
            2: (1, -1),
            3: (1, 1),
        }
        for i in range(NumClause1, NumTotClause1_2):
            state = LogicRule[i]
            a = Result[(i - NumClause1) * 2 + Numliterals1]
            b = Result[(i - NumClause1) * 2 + 1 + Numliterals1]
            if (a, b) != valid_states2.get(state):
                weight.append(weight_2SAT[state])
            else:
                W_2SAT = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
                weight.append(W_2SAT)

        # Randomly generate weights for 3SAT
        valid_states3 = {
            0: (-1, -1, -1),
            1: (-1, -1, 1),
            2: (-1, 1, -1),
            3: (-1, 1, 1),
            4: (1, -1, -1),
            5: (1, -1, 1),
            6: (1, 1, -1),
            7: (1, 1, 1),
        }
        for i in range(NumTotClause1_2, NumTotClause1_2_3):
            state = LogicRule[i]
            a, b, c = Result[(i - NumTotClause1_2) * 3 + Numliterals1_2], \
                Result[(i - NumTotClause1_2) * 3 + Numliterals1_2 + 1], \
                Result[(i - NumTotClause1_2) * 3 + Numliterals1_2 + 2]
            if (a, b, c) != valid_states3.get(state):
                weight.append(weight_3SAT[state])
            else:
                W_3SAT = [random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25),
                          random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25), random.uniform(-0.25, 0.25),
                          random.uniform(-0.125, 0.125)]
                weight.append(W_3SAT)

        # Randomly generate weights for 4SAT
        valid_states4 = {
            0: (-1, -1, -1, -1),
            1: (-1, -1, -1, 1),
            2: (-1, -1, 1, -1),
            3: (-1, -1, 1, 1),
            4: (-1, 1, -1, -1),
            5: (-1, 1, -1, 1),
            6: (-1, 1, 1, -1),
            7: (-1, 1, 1, 1),
            8: (1, -1, -1, -1),
            9: (1, -1, -1, 1),
            10: (1, -1, 1, -1),
            11: (1, -1, 1, 1),
            12: (1, 1, -1, -1),
            13: (1, 1, -1, 1),
            14: (1, 1, 1, -1),
            15: (1, 1, 1, 1),
        }
        for i in range(NumTotClause1_2_3, NumTotClause1_2_3_4):
            state = LogicRule[i]
            a, b, c, d = Result[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3], \
                Result[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 1], \
                Result[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 4], \
                Result[(i - NumTotClause1_2_3) * 10 + Numliterals1_2_3 + 5]
            if (a, b, c, d) != valid_states4.get(state):
                weight.append(weight_4SAT[state])
            else:
                W_4SAT = [random.uniform(-0.125, 0.125), random.uniform(-0.125, 0.125), random.uniform(-0.125, 0.125),
                          random.uniform(-0.125, 0.125), random.uniform(-0.125, 0.125), random.uniform(-0.125, 0.125),
                          random.uniform(-0.125, 0.125), random.uniform(-0.125, 0.125), random.uniform(-0.125, 0.125),
                          random.uniform(-0.125, 0.125), random.uniform(-0.0625, 0.0625),
                          random.uniform(-0.0625, 0.0625),
                          random.uniform(-0.0625, 0.0625), random.uniform(-0.0625, 0.0625),
                          random.uniform(-1 / 48, 1 / 48)]
                weight.append(W_4SAT)
        # Randomly generate weights for 3RedSATs
        valid_states_red3 = {
            0: (lambda a, b, c: a == -1 and b == -1),  # 0 : (A V B V C) ∧ (A V B V _C) = A V B
            1: (lambda a, b, c: a == -1 and c == -1),  # 1 : (A V B V C) ∧ (A V _B V C) = A V C
            2: (lambda a, b, c: b == -1 and c == -1),  # 2 : (A V B V C) ∧ (_A V B V C) = B V C
            3: (lambda a, b, c: a == 1 and b == 1),  # 3 : (_A V _B V _C) ∧ (_A V _B V C) = _A V _B
            4: (lambda a, b, c: a == 1 and c == 1),  # 4 : (_A V _B V _C) ∧ (_A V B V _C) = _A V _C
            5: (lambda a, b, c: b == 1 and c == 1),  # 5 : (_A V _B V _C) ∧ (A V _B V _C) = _B V _C
            6: (lambda a, b, c: a == -1 and b == 1),  # 6 : (A V _B V _C) ∧ (A V _B V C) = A V _B
            7: (lambda a, b, c: a == -1 and c == 1),  # 7 : (A V _B V _C) ∧ (A V B V _C) = A V _C
            8: (lambda a, b, c: a == 1 and b == -1),  # 8 : (_A V B V C) ∧ (_A V B V _C) = _A V B
            9: (lambda a, b, c: b == -1 and c == 1),  # 9 : (A V B V _C) ∧ (_A V B V _C) = B V _C
            10: (lambda a, b, c: a == 1 and c == 1),  # 10 : (_A V B V C) ∧ (_A V _B V C) = _A V C
            11: (lambda a, b, c: b == 1 and c == 1),  # 11 : (A V _B V C) ∧ (_A V _B V C) = _B V C
        }
        for i in range(NumTotClause1_2_3_4, NumTotClause1_2_3_4_Red3):
            state = LogicRule[i]
            a = Result[(i - NumTotClause1_2_3_4) * 3 + NewNumliterals1_2_3_4]
            b = Result[(i - NumTotClause1_2_3_4) * 3 + 1 + NewNumliterals1_2_3_4]
            c = Result[(i - NumTotClause1_2_3_4) * 3 + 2 + NewNumliterals1_2_3_4]
            condition = valid_states_red3[state]
            if not condition(a, b, c):
                weight.append(weight_3RedSAT[state])
            else:
                W_Red3SAT = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5),
                             random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
                weight.append(W_Red3SAT)
        # Randomly generate weights for 2RedSATs
        valid_states_red2 = {
            0: (lambda a, b: a == -1),  # 0 : (A V B) ∧ (A V _B) = A
            1: (lambda a, b: b == -1),  # 1 : (A V B) ∧ (_A V B) = B
            2: (lambda a, b: a == 1),  # 2 : (_A V _B) ∧ (_A V B) = _A
            3: (lambda a, b: b == 1),  # 3 : (_A V _B) ∧ (A V _B) = _B
        }
        for i in range(NumTotClause1_2_3_4_Red3, NumTotClause1_2_3_4_Red3_Red2):
            state = LogicRule[i]
            a = Result[(i - NumTotClause1_2_3_4_Red3) * 2 + Numliterals1_2_3_4_Red3]
            b = Result[(i - NumTotClause1_2_3_4_Red3) * 2 + 1 + Numliterals1_2_3_4_Red3]
            condition = valid_states_red2[state]
            if not condition(a, b):
                weight.append(weight_2RedSAT[state])
            else:
                W_Red2SAT = [random.uniform(-1, 1), random.uniform(-1, 1)]
                weight.append(W_Red2SAT)
        # valid states for MAX2SAT
        valid_states_MAX2SAT = {
            0: {(1, -1, -1), (-1, 1, 1)},
            1: {(1, -1, 1), (-1, 1, -1)},
            2: {(1, -1, 1), (-1, 1, -1)},
            3: {(1, -1, -1), (-1, 1, 1)},
        }
        for i in range(NumTotClause1_2_3_4_Red3_Red2, NumTotClause_MAX2SAT):
            state = LogicRule[i]
            a = Result[(i - NumTotClause1_2_3_4_Red3_Red2) * 2 + Numliterals1_2_3_4_Red3_Red2]
            b = Result[(i - NumTotClause1_2_3_4_Red3_Red2) * 2 + 1 + Numliterals1_2_3_4_Red3_Red2]
            c = Result[(i - NumTotClause1_2_3_4_Red3_Red2) * 2 + 2 + Numliterals1_2_3_4_Red3_Red2]
            allowed = valid_states_MAX2SAT[state]
            if (a, b, c) not in allowed:
                weight.append(weight_MAX2SAT[state])
            else:
                W_MAX2SAT = [0, 0, random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
                weight.append(W_MAX2SAT)


#  4.2 Calculate the Minimum Energy
def calculate_minimum_energy(NumClause1, NumClause2, NumClause3, NumClause4, NumRedClause3, NumRedClause2, NumRedClauseMAX2SAT):
    return - (1 / 2) * NumClause1 - (1 / 4) * NumClause2 - (1 / 8) * NumClause3 - (1 / 16) * NumClause4 - (
            1 / 4) * NumRedClause3 - (1 / 2) * NumRedClause2 - (1 / 4) * NumRedClauseMAX2SAT


#  4.3 Calculate local field and update neuron states
def S_chose(x):
    if x > 0:
        return 1
    elif x == 0:
        return x
    elif x < 0:
        return -1
    return None


# calculate_local_field, update_neurons, calculate final energy, calculate similarity
def calculate_neurons(relaxation_num, NumClauseList_Used, NumLiteralList_Used, Maxcomb, NeuronString, valid_states4,
                      All_weights, NumTrail, Minimum_energy, Tol, LogicRule):
    Energy_list = [[] for _ in range(Maxcomb)]
    # Different Ratios
    Num_Solved_SAT = []
    Ratio_Solved_Component = []
    Ratio_Solved_Initial = []
    Ratio_Solved_Initial_Satisfied = []
    Ratio_Solved_Red = []
    Ratio_Solved_Red_Satisfied = []
    Ratio_Solved_Tse = []
    Ratio_Solved_Tse_Satisfied = []
    Ratio_Solved_After = []
    Ratio_Solved_After_Satisfied = []
    # Similarity_Ablation
    Similarity_list_J_Initial = []
    Similarity_list_H_Initial = []
    Similarity_list_J_Red = []
    Similarity_list_H_Red = []
    Similarity_list_J_Tse = []
    Similarity_list_H_Tse = []
    Similarity_list_J_After = []
    Similarity_list_H_After = []

    NumGlobal_list = []
    T_TheSetOfTest = []
    # 1. Perform 100 tests on each combination
    for j in range(Maxcomb):  # the number of len(All_weights) is 100
        Numliterals1_2 = NumLiteralList_Used[j][1]
        Numliterals1_2_3 = NumLiteralList_Used[j][2]
        NewNumliterals1_2_3_4 = NumLiteralList_Used[j][4]
        Numliterals1_2_3_4_Red3 = NumLiteralList_Used[j][5]
        Numliterals1_2_3_4_Red3_Red2 = NumLiteralList_Used[j][6]
        TraNumliterals = NumLiteralList_Used[j][7]

        NumClause1 = NumClauseList_Used[j][0]
        NumClause1_2 = NumClauseList_Used[j][1]
        NumClause1_2_3 = NumClauseList_Used[j][2]
        NumClause1_2_3_4 = NumClauseList_Used[j][3]
        NumClause1_2_3_4_Red3 = NumClauseList_Used[j][4]
        NumClause1_2_3_4_Red3_Red2 = NumClauseList_Used[j][5]
        NumTotClause = NumClauseList_Used[j][6]
        # Generate neuron states to test
        TheSetOfTest = np.zeros((TraNumliterals, NumTrail), dtype=int)
        generate_retrieval_neuron_state(TraNumliterals, NeuronString, TheSetOfTest)
        # print('TheNeuronState', TheSetOfTest, TheSetOfTest.shape)
        indicator = 1
        # 2. the number of relaxation
        for m in range(relaxation_num):
            for l in range(NumTrail):
                Energy = 0
                gradient = 0
                for k in range(len(All_weights[j])):
                    if len(All_weights[j][k]) == 1:
                        h_1_1 = All_weights[j][k][0]
                        ns1 = S_chose(np.tanh(h_1_1))
                        TheSetOfTest[gradient][l] = ns1 if ns1 == -1 or ns1 == 1 else TheSetOfTest[gradient][l]
                        if indicator == relaxation_num:  # judging if we need to calculate energy
                            Energy += -All_weights[j][k][0] * TheSetOfTest[gradient][l]
                            # print('1SAT', Energy)
                        else:
                            pass
                        gradient += 1

                    elif len(All_weights[j][k]) == 3:
                        h_2_1 = All_weights[j][k][2] * TheSetOfTest[gradient + 1][l] + All_weights[j][k][0]
                        ns1 = S_chose(np.tanh(h_2_1))
                        TheSetOfTest[gradient][l] = ns1 if ns1 == -1 or ns1 == 1 else TheSetOfTest[gradient][l]

                        h_2_2 = All_weights[j][k][2] * TheSetOfTest[gradient][l] + All_weights[j][k][1]
                        ns2 = S_chose(np.tanh(h_2_2))
                        TheSetOfTest[gradient + 1][l] = ns2 if ns2 == -1 or ns2 == 1 else TheSetOfTest[gradient + 1][l]
                        if indicator == relaxation_num:  # judging if we need to calculate energy
                            Energy += -All_weights[j][k][2] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 1][
                                l] - \
                                      All_weights[j][k][1] * TheSetOfTest[gradient + 1][l] - All_weights[j][k][0] * \
                                      TheSetOfTest[gradient][l]
                            # print('2SAT', Energy)
                        else:
                            pass
                        gradient += 2
                    # 0 : A V B V C    # 1 : A V B V _C    # 2 : A V _B V C    # 3 : A V _B V _C
                    # 4 : _A V B V C    # 5 : _A V  B V _C    # 6 : _A V _B V C    # 7 : _A V _B V _C
                    elif len(All_weights[j][k]) == 7:
                        h_3_1 = 2 * All_weights[j][k][6] * TheSetOfTest[gradient + 1][l] * \
                                TheSetOfTest[gradient + 2][
                                    l] + \
                                All_weights[j][k][4] * TheSetOfTest[gradient + 2][l] + All_weights[j][k][3] * \
                                TheSetOfTest[gradient + 1][l] + \
                                All_weights[j][k][0]
                        ns1 = S_chose(np.tanh(h_3_1))
                        TheSetOfTest[gradient][l] = ns1 if ns1 == -1 or ns1 == 1 else TheSetOfTest[gradient][l]

                        h_3_2 = 2 * All_weights[j][k][6] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 2][
                            l] + \
                                All_weights[j][k][5] * TheSetOfTest[gradient + 2][l] + All_weights[j][k][3] * \
                                TheSetOfTest[gradient][l] + \
                                All_weights[j][k][1]
                        ns2 = S_chose(np.tanh(h_3_2))
                        TheSetOfTest[gradient + 1][l] = ns2 if ns2 == -1 or ns2 == 1 else TheSetOfTest[gradient + 1][l]

                        h_3_3 = 2 * All_weights[j][k][6] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 1][
                            l] + \
                                All_weights[j][k][5] * TheSetOfTest[gradient + 1][l] + All_weights[j][k][4] * \
                                TheSetOfTest[gradient][l] + \
                                All_weights[j][k][2]
                        ns3 = S_chose(np.tanh(h_3_3))
                        TheSetOfTest[gradient + 2][l] = ns3 if ns3 == -1 or ns3 == 1 else TheSetOfTest[gradient + 2][l]
                        if indicator == relaxation_num:  # judging if we need to calculate energy
                            Energy += -2 * All_weights[j][k][6] * TheSetOfTest[gradient][l] * \
                                      TheSetOfTest[gradient + 1][
                                          l] * TheSetOfTest[gradient + 2][l] - \
                                      All_weights[j][k][5] * TheSetOfTest[gradient + 1][l] * TheSetOfTest[gradient + 2][
                                          l] - \
                                      All_weights[j][k][4] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 2][l] - \
                                      All_weights[j][k][3] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 1][l] - \
                                      All_weights[j][k][2] * TheSetOfTest[gradient + 2][l] - All_weights[j][k][1] * \
                                      TheSetOfTest[gradient + 1][l] - \
                                      All_weights[j][k][0] * TheSetOfTest[gradient][l]
                            # print('3SAT', Energy)
                        else:
                            pass
                        gradient += 3
                        # 0 : A V B V C V D      # 1 : A V B V C V _D    # 2 : A V B V _C V D
                        # 3 : A V B V _C V _D    # 4 : A V _B V C V D    # 5 : A V _B V C V _D
                        # 6 : A V _B V _C V D    # 7 :A V _B V _C V _D    # 8 : _A V B V C V D
                        # 9 : _A V B V C V _D    # 10 : _A V B V _C V D    # 11 : _A V B V _C V _D
                        # 12 : _A V _B V C V D    # 13 : _A V _B V C V _D    # 14 : _A V _B V _C V D  # 15 : _A V _B V _C V _D

                    elif len(All_weights[j][k]) == 15:
                        h_4_1 = 6 * All_weights[j][k][14] * TheSetOfTest[gradient + 1][l] * \
                                TheSetOfTest[gradient + 4][
                                    l] * \
                                TheSetOfTest[gradient + 5][l] + \
                                2 * All_weights[j][k][12] * TheSetOfTest[gradient + 4][l] * \
                                TheSetOfTest[gradient + 5][
                                    l] + \
                                2 * All_weights[j][k][11] * TheSetOfTest[gradient + 1][l] * \
                                TheSetOfTest[gradient + 5][
                                    l] + \
                                2 * All_weights[j][k][10] * TheSetOfTest[gradient + 1][l] * \
                                TheSetOfTest[gradient + 4][
                                    l] + \
                                All_weights[j][k][6] * TheSetOfTest[gradient + 5][l] + All_weights[j][k][5] * \
                                TheSetOfTest[gradient + 4][l] + \
                                All_weights[j][k][4] * TheSetOfTest[gradient + 1][l] + All_weights[j][k][0]
                        ns1 = S_chose(np.tanh(h_4_1))
                        TheSetOfTest[gradient][l] = ns1 if ns1 == -1 or ns1 == 1 else TheSetOfTest[gradient][l]

                        h_4_2 = 6 * All_weights[j][k][14] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 4][
                            l] * \
                                TheSetOfTest[gradient + 5][l] + 2 * All_weights[j][k][13] * \
                                TheSetOfTest[gradient + 4][l] * TheSetOfTest[gradient + 5][l] + 2 * \
                                All_weights[j][k][
                                    11] * \
                                TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 5][l] + 2 * All_weights[j][k][
                                    10] * \
                                TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 4][l] + All_weights[j][k][8] * \
                                TheSetOfTest[gradient + 5][l] + All_weights[j][k][7] * TheSetOfTest[gradient + 4][
                                    l] + \
                                All_weights[j][k][4] * TheSetOfTest[gradient][l] + All_weights[j][k][1]
                        ns2 = S_chose(np.tanh(h_4_2))
                        TheSetOfTest[gradient + 1][l] = ns2 if ns2 == -1 or ns2 == 1 else \
                            TheSetOfTest[gradient + 1][l]

                        h_4_5 = 6 * All_weights[j][k][14] * TheSetOfTest[gradient][l] * \
                                TheSetOfTest[gradient + 1][l] * TheSetOfTest[gradient + 5][l] + 2 * \
                                All_weights[j][k][
                                    13] * \
                                TheSetOfTest[gradient + 1][l] * TheSetOfTest[gradient + 5][l] + 2 * \
                                All_weights[j][k][
                                    12] * \
                                TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 5][l] + 2 * All_weights[j][k][
                                    10] * \
                                TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 1][l] + All_weights[j][k][9] * \
                                TheSetOfTest[gradient + 5][l] + All_weights[j][k][7] * TheSetOfTest[gradient + 1][
                                    l] + \
                                All_weights[j][k][5] * TheSetOfTest[gradient][l] + All_weights[j][k][2]
                        ns5 = S_chose(np.tanh(h_4_5))
                        TheSetOfTest[gradient + 4][l] = ns5 if ns5 == -1 or ns5 == 1 else \
                            TheSetOfTest[gradient + 4][l]

                        h_4_6 = 6 * All_weights[j][k][14] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 1][
                            l] * \
                                TheSetOfTest[gradient + 4][l] + 2 * All_weights[j][k][13] * \
                                TheSetOfTest[gradient + 1][l] * TheSetOfTest[gradient + 4][l] + 2 * \
                                All_weights[j][k][
                                    12] * \
                                TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 4][l] + 2 * All_weights[j][k][
                                    11] * \
                                TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 1][l] + All_weights[j][k][9] * \
                                TheSetOfTest[gradient + 4][l] + All_weights[j][k][8] * TheSetOfTest[gradient + 1][
                                    l] + \
                                All_weights[j][k][6] * TheSetOfTest[gradient][l] + All_weights[j][k][3]
                        ns6 = S_chose(np.tanh(h_4_6))
                        TheSetOfTest[gradient + 5][l] = ns6 if ns6 == -1 or ns6 == 1 else \
                            TheSetOfTest[gradient + 5][l]

                        # conversion process
                        state = LogicRule[j][k]
                        a = TheSetOfTest[gradient][l]
                        b = TheSetOfTest[gradient + 1][l]
                        if (a, b) != valid_states4.get(state):
                            TheSetOfTest[gradient + 2][l] = 1
                            TheSetOfTest[gradient + 3][l] = 1
                        else:
                            TheSetOfTest[gradient + 2][l] = -1
                            TheSetOfTest[gradient + 3][l] = -1

                        TheSetOfTest[gradient + 6][l] = a
                        TheSetOfTest[gradient + 7][l] = TheSetOfTest[gradient + 3][l]
                        TheSetOfTest[gradient + 8][l] = b
                        TheSetOfTest[gradient + 9][l] = TheSetOfTest[gradient + 3][l]

                        if indicator == relaxation_num:  # judging if we need to calculate energy
                            Energy += (-6 * All_weights[j][k][14] * TheSetOfTest[gradient][l] *
                                       TheSetOfTest[gradient + 1][l] * TheSetOfTest[gradient + 4][l] *
                                       TheSetOfTest[gradient + 5][l] - 2 * All_weights[j][k][13] *
                                       TheSetOfTest[gradient + 1][l] * TheSetOfTest[gradient + 4][l] *
                                       TheSetOfTest[gradient + 5][l] -
                                       2 * All_weights[j][k][12] * TheSetOfTest[gradient][l] *
                                       TheSetOfTest[gradient + 4][
                                           l] * TheSetOfTest[gradient + 5][l] -
                                       2 * All_weights[j][k][11] * TheSetOfTest[gradient][l] *
                                       TheSetOfTest[gradient + 1][
                                           l] * TheSetOfTest[gradient + 5][l] -
                                       2 * All_weights[j][k][10] * TheSetOfTest[gradient][l] *
                                       TheSetOfTest[gradient + 1][
                                           l] * TheSetOfTest[gradient + 4][l] -
                                       All_weights[j][k][9] * TheSetOfTest[gradient + 4][l] *
                                       TheSetOfTest[gradient + 5][
                                           l] -
                                       All_weights[j][k][8] * TheSetOfTest[gradient + 1][l] *
                                       TheSetOfTest[gradient + 5][
                                           l] -
                                       All_weights[j][k][7] * TheSetOfTest[gradient + 1][l] *
                                       TheSetOfTest[gradient + 4][
                                           l] -
                                       All_weights[j][k][6] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 5][
                                           l] -
                                       All_weights[j][k][5] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 4][
                                           l] -
                                       All_weights[j][k][4] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 1][
                                           l] -
                                       All_weights[j][k][3] * TheSetOfTest[gradient + 5][l] - All_weights[j][k][2] *
                                       TheSetOfTest[gradient + 4][l] -
                                       All_weights[j][k][1] * TheSetOfTest[gradient + 1][l] - All_weights[j][k][0] *
                                       TheSetOfTest[gradient][l])
                            # print('4SAT', Energy)
                        else:
                            pass
                        gradient += 10

                    elif len(All_weights[j][k]) == 6:
                        h_red3_1 = All_weights[j][k][4] * TheSetOfTest[gradient + 2][l] + \
                                   All_weights[j][k][3] * TheSetOfTest[gradient + 1][l] + All_weights[j][k][0]
                        ns1 = S_chose(np.tanh(h_red3_1))
                        TheSetOfTest[gradient][l] = ns1 if ns1 == -1 or ns1 == 1 else TheSetOfTest[gradient][l]

                        h_red3_2 = All_weights[j][k][5] * TheSetOfTest[gradient + 2][l] + \
                                   All_weights[j][k][3] * TheSetOfTest[gradient][l] + All_weights[j][k][1]
                        ns2 = S_chose(np.tanh(h_red3_2))
                        TheSetOfTest[gradient + 1][l] = ns2 if ns2 == -1 or ns2 == 1 else \
                            TheSetOfTest[gradient + 1][l]

                        h_red3_3 = All_weights[j][k][5] * TheSetOfTest[gradient + 1][l] + \
                                   All_weights[j][k][4] * TheSetOfTest[gradient][l] + All_weights[j][k][2]
                        ns3 = S_chose(np.tanh(h_red3_3))
                        TheSetOfTest[gradient + 2][l] = ns3 if ns3 == -1 or ns3 == 1 else \
                            TheSetOfTest[gradient + 2][l]

                        if indicator == relaxation_num:  # judging if we need to calculate energy
                            Energy += -All_weights[j][k][5] * TheSetOfTest[gradient + 1][l] * \
                                      TheSetOfTest[gradient + 2][
                                          l] - \
                                      All_weights[j][k][4] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 2][l] - \
                                      All_weights[j][k][3] * TheSetOfTest[gradient + 1][l] * TheSetOfTest[gradient][l] - \
                                      All_weights[j][k][2] * TheSetOfTest[gradient + 2][l] - All_weights[j][k][1] * \
                                      TheSetOfTest[gradient + 1][l] - All_weights[j][k][0] * TheSetOfTest[gradient][l]
                            # print('Red3SAT', Energy)
                        else:
                            pass
                        gradient += 3
                        # 0 : (A V B) ∧ (A V _B) = A    # 1 : (A V B) ∧ (_A V B) = B
                        # 2 : (_A V _B) ∧ (_A V B) = _A    # 3 : (_A V _B) ∧ (A V _B) = _B

                    elif len(All_weights[j][k]) == 2:
                        h_red2_1 = All_weights[j][k][0]
                        ns1 = S_chose(np.tanh(h_red2_1))
                        TheSetOfTest[gradient][l] = ns1 if ns1 == -1 or ns1 == 1 else TheSetOfTest[gradient][l]

                        h_red2_2 = All_weights[j][k][1]
                        ns2 = S_chose(np.tanh(h_red2_2))
                        TheSetOfTest[gradient + 1][l] = ns2 if ns2 == -1 or ns2 == 1 else \
                            TheSetOfTest[gradient + 1][l]
                        if indicator == relaxation_num:  # judging if we need to calculate energy
                            Energy += - All_weights[j][k][1] * \
                                      TheSetOfTest[gradient + 1][l] - All_weights[j][k][0] * TheSetOfTest[gradient][l]
                            # print('Red2SAT', Energy)
                        else:
                            pass
                        gradient += 2

                        # 0 : (_p V _q) ∧ (_p V q) ∧ (p V _q) ∧ (p V _r) ∧ (q V r)     # 1 : (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (q V _r) ∧ (p V r)
                        # 2: (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_p V _r) ∧ (_q V r)      # 3 : (p V q) ∧ (_p V q) ∧ (p V _q) ∧ (_q V _r) ∧ (_p V r)
                    elif len(All_weights[j][k]) == 5:
                        h_MAX2SAT_1 = All_weights[j][k][2] * TheSetOfTest[gradient + 1][l] + All_weights[j][k][3] * \
                                      TheSetOfTest[gradient + 2][l]
                        ns1 = S_chose(np.tanh(h_MAX2SAT_1))
                        TheSetOfTest[gradient][l] = ns1 if ns1 == -1 or ns1 == 1 else TheSetOfTest[gradient][l]

                        h_MAX2SAT_2 = All_weights[j][k][2] * TheSetOfTest[gradient][l] + All_weights[j][k][4] * \
                                      TheSetOfTest[gradient + 2][l]
                        ns2 = S_chose(np.tanh(h_MAX2SAT_2))
                        TheSetOfTest[gradient + 1][l] = ns2 if ns2 == -1 or ns2 == 1 else TheSetOfTest[gradient + 1][l]

                        h_MAX2SAT_3 = All_weights[j][k][3] * TheSetOfTest[gradient][l] + All_weights[j][k][4] * \
                                      TheSetOfTest[gradient + 1][l]
                        ns2 = S_chose(np.tanh(h_MAX2SAT_3))
                        TheSetOfTest[gradient + 2][l] = ns2 if ns2 == -1 or ns2 == 1 else TheSetOfTest[gradient + 2][l]

                        if indicator == relaxation_num:  # judging if we need to calculate energy
                            Energy += (- All_weights[j][k][2] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 1][
                                l]
                                       - All_weights[j][k][3] * TheSetOfTest[gradient][l] * TheSetOfTest[gradient + 2][
                                           l]
                                       - All_weights[j][k][4] * TheSetOfTest[gradient + 1][l] *
                                       TheSetOfTest[gradient + 2][l])
                            # print('MAX2SAT', Energy)
                        else:
                            pass
                        gradient += 3

                if indicator == relaxation_num:  # judging if we need to calculate energy
                    Energy_list[j].append(Energy)  # save the value of energy in list
                else:
                    pass
            indicator += 1
        # 3. SAT_Solver

        # Save Different Ratios
        TheFitnessNumOri = []
        TheRatioFitnessNumOri = []
        InitialRatio = []  #
        InitialRatioSatisfied = []  #
        RedRatio = []  #
        RedRatioSatisfied = []  #
        TseRatio = []  #
        TseRatioSatisfied = []  #
        AfterRatio = []  #
        AfterRatioSatisfied = []  #
        for k in range(NeuronString):
            FitnessNumOri, RatioFitnessNumOri, InitialRatioFitnessNum, InitialRatioFitnessNumSatisfied, RedRatioFitnessNum, RedRatioFitnessNumSatisfied, \
                TseRatioFitnessNum, TseRatioFitnessNumSatisfied, AfterRatioFitnessNum, AfterRatioFitnessNumSatisfied = calculate_fitness(k, NumClause1, NumClause1_2, NumClause1_2_3, NumClause1_2_3_4, NumClause1_2_3_4_Red3,
                                                                                                                                         NumClause1_2_3_4_Red3_Red2, NumTotClause, Numliterals1_2, Numliterals1_2_3, NewNumliterals1_2_3_4,
                                                                                                                                         Numliterals1_2_3_4_Red3, Numliterals1_2_3_4_Red3_Red2, LogicRule[j], TheSetOfTest)

            # Save Different Ratios
            TheFitnessNumOri.append(FitnessNumOri)
            TheRatioFitnessNumOri.append(RatioFitnessNumOri)  # Ratio of components
            InitialRatio.append(InitialRatioFitnessNum)
            InitialRatioSatisfied.append(InitialRatioFitnessNumSatisfied)
            RedRatio.append(RedRatioFitnessNum)
            RedRatioSatisfied.append(RedRatioFitnessNumSatisfied)
            TseRatio.append(TseRatioFitnessNum)
            TseRatioSatisfied.append(TseRatioFitnessNumSatisfied)
            AfterRatio.append(AfterRatioFitnessNum)
            AfterRatioSatisfied.append(AfterRatioFitnessNumSatisfied)

        # 4. For each logica rule, save the number of Fitness/Ratio [, ,, , ,, , , ,]
        TheNotFitnessNum = [len(All_weights[j]) - x for x in TheFitnessNumOri]
        Num_Solved_SAT.append(TheNotFitnessNum)
        Ratio_Solved_Component.append(TheRatioFitnessNumOri)  # Ratio of components
        Ratio_Solved_Initial.append(InitialRatio)
        Ratio_Solved_Initial_Satisfied.append(InitialRatioSatisfied)
        Ratio_Solved_Red.append(RedRatio)
        Ratio_Solved_Red_Satisfied.append(RedRatioSatisfied)
        Ratio_Solved_Tse.append(TseRatio)
        Ratio_Solved_Tse_Satisfied.append(TseRatioSatisfied)
        Ratio_Solved_After.append(AfterRatio)
        Ratio_Solved_After_Satisfied.append(AfterRatioSatisfied)
        # print(TheRatioFitnessNumRed)
        # 5. Save Final neuron state
        New_TheSetOfTest = np.array(TheSetOfTest).T
        T_TheSetOfTest.append(New_TheSetOfTest)
    # print(Minimum_energy)
    # print(np.array(Energy_list).shape, Energy_list)
    # 3.0 SAT_Solver
    '''
    print(np.array(Num_Solved_SAT).shape, 'The number of clauses not solved:', Num_Solved_SAT)
    print('np.mean(Num_Solved_SAT):', np.mean(Num_Solved_SAT), 'and', 'np.var(Num_Solved_SAT):', np.var(Num_Solved_SAT))
    '''
    Solved_SAT_Count = 0
    Z_SATSolver_index_list = []
    for i in range(Maxcomb):
        Z_SATSolver_index_list.append(min(Num_Solved_SAT[i]))
        for j in range(NumTrail):
            if Num_Solved_SAT[i][j] == 0:
                Solved_SAT_Count += 1
            else:
                # print(LogicRule[i])
                # print(T_TheSetOfTest[i][j])
                pass
    # A save SAT
    # A_SAT_list = [np.mean(Z_SATSolver_index_list), Solved_SAT_Count / (Maxcomb * NumTrail), 1 - Solved_SAT_Count / (Maxcomb * NumTrail), np.mean(Num_Solved_SAT), np.std(Num_Solved_SAT)]
    #
    A_SAT_list = [np.mean(Z_SATSolver_index_list), np.std(Z_SATSolver_index_list), np.mean(Num_Solved_SAT), np.std(Num_Solved_SAT), Solved_SAT_Count / (Maxcomb * NumTrail)]
    #
    A_Ratio_SAT_list = [np.mean(Ratio_Solved_Component), np.mean(Ratio_Solved_Initial), np.mean(Ratio_Solved_Initial_Satisfied), np.mean(Ratio_Solved_Red), np.mean(Ratio_Solved_Red_Satisfied), np.mean(Ratio_Solved_Tse),
                        np.mean(Ratio_Solved_Tse_Satisfied), np.mean(Ratio_Solved_After), np.mean(Ratio_Solved_After_Satisfied)]
    '''
    print('The ratio of solved SAT:', Solved_SAT_Count / (Maxcomb * NumTrail))
    print('The ratio of not solved SAT:', 1 - Solved_SAT_Count / (Maxcomb * NumTrail))
    print('-----------------------------------------------------------------------------------')
    '''
    # 6. Calculate Final Energy Error
    Total_Energy_in_same_neurons = []  # Perform NumTrail/100 tests on each combination
    for n in range(Maxcomb):
        Energy_Error = 0
        for o in range(NumTrail):
            Energy_Error += abs(Energy_list[n][o] - Minimum_energy[n])
        Total_Energy_in_same_neurons.append(Energy_Error)
    MAE_Energy = np.mean(Total_Energy_in_same_neurons) / Maxcomb
    '''
    print('Error_Energy_in_same_neurons:', Total_Energy_in_same_neurons)
    print('np.max(Total_Energy_in_same_neurons)', np.max(Total_Energy_in_same_neurons))
    print('MAE_Energy:', MAE_Energy)
    '''
    # 6. Calculate Ratio of global minimum solutions
    if MAE_Energy == 0:
        ## print('Ratio of local minimum solutions:', 0)
        ## print('Ratio of global minimum solutions:', 1)
        B_Energy_list = [MAE_Energy, 1, 0]
    else:
        TheNumLocal = 0
        for p in range(Maxcomb):
            NumLocal = 0
            for q in range(NumTrail):
                if abs(Energy_list[p][q] - Minimum_energy[p]) > Tol:
                    NumLocal += 1
                else:
                    pass
            NumGlobal_list.append(NumTrail - NumLocal)
            TheNumLocal += NumLocal
        ## print('NumGlobal_list:', NumGlobal_list)
        ## print('np.max(NumGloab_list):', np.max(NumGlobal_list))
        Z_l = TheNumLocal / (Maxcomb * NumTrail)
        ## print('Ratio of local minimum solutions:', Z_l)
        ## print('Ratio of global minimum solutions:', 1 - Z_l)
        # B saved energy
        B_Energy_list = [MAE_Energy, 1 - Z_l, Z_l]
    ## print('-----------------------------------------------------------------------------------')
    # 7. calculate the similarity
    for p in range(Maxcomb):
        Similarity_count_J_Initial = 0
        Similarity_count_H_Initial = 0
        Similarity_count_J_Red = 0
        Similarity_count_H_Red = 0
        Similarity_count_J_Tse = 0
        Similarity_count_H_Tse = 0
        Similarity_count_J_After = 0
        Similarity_count_H_After = 0
        NumEffectiveTrail = 0
        NumCount = 0
        InitialLiteralNum = NumLiteralList_Used[p][3] + NumClauseList_Used[p][4] * 6 + NumClauseList_Used[p][5] * 4 + NumClauseList_Used[p][6] * 10
        RedLiteralNum = NumLiteralList_Used[p][3] + NumClauseList_Used[p][4] * 2 + NumClauseList_Used[p][5] + NumClauseList_Used[p][6] * 10
        TseLiteralNum = NumLiteralList_Used[p][2] + NumClauseList_Used[p][3] * 10 + NumClauseList_Used[p][4] * 6 + NumClauseList_Used[p][5] * 4 + NumClauseList_Used[p][6] * 10
        AfterLiteralNum = NumLiteralList_Used[p][2] + NumClauseList_Used[p][3] * 10 + NumClauseList_Used[p][4] * 2 + NumClauseList_Used[p][5] + NumClauseList_Used[p][6] * 10
        for q in range(NumTrail):
            SIM_J_Initial, SIM_H_Initial, SIM_J_Red, SIM_H_Red, SIM_J_Tse, SIM_H_Tse, SIM_J_After, SIM_H_After = check_similarity(LogicRule[p], T_TheSetOfTest[p][q], NumLiteralList_Used[p][0], NumLiteralList_Used[p][1],
                                                                                                                                  NumLiteralList_Used[p][2], NumLiteralList_Used[p][4],
                                                                                                                                  NumLiteralList_Used[p][5], NumLiteralList_Used[p][6], NumLiteralList_Used[p][7], NumClauseList_Used[p][0],
                                                                                                                                  NumClauseList_Used[p][1], NumClauseList_Used[p][2], NumClauseList_Used[p][3], NumClauseList_Used[p][4],
                                                                                                                                  NumClauseList_Used[p][5])
            if Num_Solved_SAT[p][q] == 0:
                NumEffectiveTrail += 1
                Similarity_count_J_Initial += SIM_J_Initial
                Similarity_count_H_Initial += SIM_H_Initial
                Similarity_count_J_Red += SIM_J_Red
                Similarity_count_H_Red += SIM_H_Red
                Similarity_count_J_Tse += SIM_J_Tse
                Similarity_count_H_Tse += SIM_H_Tse
                Similarity_count_J_After += SIM_J_After
                Similarity_count_H_After += SIM_H_After
        NumCount += NumEffectiveTrail
        if NumCount != 0:
            """
            Similarity_list_J_Initial.append(1)
            Similarity_list_H_Initial.append(0)
            Similarity_list_J_Red.append(1)
            Similarity_list_H_Red.append(0)
            Similarity_list_J_Tse.append(1)
            Similarity_list_H_Tse.append(0)
            Similarity_list_J_After.append(1)
            Similarity_list_H_After.append(0)
        else:
        """
            Similarity_list_J_Initial.append(Similarity_count_J_Initial / NumCount)
            Similarity_list_H_Initial.append(Similarity_count_H_Initial / (NumCount * InitialLiteralNum))
            Similarity_list_J_Red.append(Similarity_count_J_Red / NumCount)
            Similarity_list_H_Red.append(Similarity_count_H_Red / (NumCount * RedLiteralNum))
            Similarity_list_J_Tse.append(Similarity_count_J_Tse / NumCount)
            Similarity_list_H_Tse.append(Similarity_count_H_Tse / (NumCount * TseLiteralNum))
            Similarity_list_J_After.append(Similarity_count_J_After / NumCount)
            Similarity_list_H_After.append(Similarity_count_H_After / (NumCount * AfterLiteralNum))

    C_Similarity_list = [np.mean(Similarity_list_J_Initial), np.std(Similarity_list_J_Initial), np.mean(Similarity_list_H_Initial), np.std(Similarity_list_H_Initial)]
    C_Similarity_list_ablation = [np.mean(Similarity_list_J_Red), np.std(Similarity_list_J_Red), np.mean(Similarity_list_H_Red), np.std(Similarity_list_H_Red),
                                  np.mean(Similarity_list_J_Tse), np.std(Similarity_list_J_Tse), np.mean(Similarity_list_H_Tse), np.std(Similarity_list_H_Tse),
                                  np.mean(Similarity_list_J_After), np.std(Similarity_list_J_After), np.mean(Similarity_list_H_After), np.std(Similarity_list_H_After)]
    return A_SAT_list, A_Ratio_SAT_list, B_Energy_list, C_Similarity_list, C_Similarity_list_ablation


# random iteration by time
def reseed_random():
    seed = int(time.time() * 1000000) % (2 ** 32 - 1)
    np.random.seed(seed)
    random.seed(seed)


# calculate learning_error
def calculate_Error(index_max, FitnessNum, NumTotClause, TheMAE, TheRMSE, TheMAPE):
    # Calculate Errors
    Error_MAE = 0
    Error_RMSE = 0
    Error_MAPE = 0
    for l in range(index_max + 1):
        Error = NumTotClause - FitnessNum[l]
        Error_MAE += Error
        Error_RMSE += Error ** 2
        Error_MAPE += abs(NumTotClause - FitnessNum[l]) / NumTotClause
    MAE_Error = Error_MAE / (index_max + 1)
    RMSE_Error = math.sqrt(Error_RMSE / (index_max + 1))
    MAPE_Error = 100 * Error_MAPE / (index_max + 1)

    TheMAE.append(MAE_Error)
    TheRMSE.append(RMSE_Error)
    TheMAPE.append(MAPE_Error)


def main():
    #  1. The Main Function of Learning Phase
    MaxComb = 100  # 100 Max combination of Logic rule
    NumLearn = 100  # The Number of learning
    NumTrail = 100  # The Number of trails
    NeuronString = 100  # The Number of neuron states
    print(" DHNN-MAX2SAT  \n")
    print(f" Number of Neuron Combination = {NeuronString}")
    print(f" Number of Trial = {NumTrail}")
    print(f" Number of Learning = {NumLearn} \n")

    MAE_list = []
    RMSE_list = []
    MAPE_list = []
    Standard_MAE_Learn = []
    Standard_RMSE_Learn = []
    Standard_MAPE_Learn = []
    All_weights = []
    All_LogicRules = []
    Minimum_energy_list = []
    TheInformationOfClause = []
    TheInformationOfLiteral = []
    TheInformationOfClause_Used = []
    TheInformationOfLiteral_Used = []
    reseed_random()

    # generate logic rules in fixed number of neuron
    for i in range(6, 6 * 21, 6):
        NUMNEURONS = i
        TheNumClauseList = []
        TheNumLiteralList = []
        TheNumClauseList_Used = []
        TheNumLiteralList_Used = []
        TheLogicRule_list = []
        Minimum_energy = []
        MAE = []
        RMSE = []
        MAPE = []
        The_weight = [[] for _ in range(MaxComb)]
        Test_list = []
        # 'MAX2SAT', 'MAX2SAT0.2', 'MAX2SAT0.4', 'MAX2SAT0.6', 'MAX2SAT0.8', 'MAX2SAT1'
        SATtype = 0
        Solutions = Combination(NUMNEURONS, SATtype)
        for j in range(MaxComb):
            # 0.1 Randomly generated clause quantity
            TheNumClause1, TheNumClause2, TheNumClause3, TheNumClause4, TheNumRedClause3, TheNumRedClause2, TheNumClauseMAX2SAT = simple_random_solution(Solutions, SATtype)
            # 0.2 Calculate the minimum_energy
            Minimum_energy.append(calculate_minimum_energy(TheNumClause1, TheNumClause2, TheNumClause3, TheNumClause4, TheNumRedClause3, TheNumRedClause2, TheNumClauseMAX2SAT))
            # 0.3 Calculate the number of literals
            TheNumliterals1 = TheNumClause1
            TheNumliterals2 = TheNumClause2 * 2
            TheNumliterals3 = TheNumClause3 * 3
            TheNumliterals4 = TheNumClause4 * 4
            TheNumRedliterals3 = TheNumRedClause3 * 3
            TheNumRedliterals2 = TheNumRedClause2 * 2
            TheNumliteralsMAX2SAT = TheNumClauseMAX2SAT * 3
            # 0.4 Calculate the number of used literals
            TheNumliterals1_2 = TheNumliterals1 + TheNumliterals2
            TheNumliterals1_2_3 = TheNumliterals1_2 + TheNumliterals3
            TheNumliterals1_2_3_4 = TheNumliterals1_2_3 + TheNumliterals4
            TheNewNumliterals1_2_3_4 = TheNumliterals1_2_3_4 + TheNumClause4 * 6  # After 4-SAT Conversion To 3-SAT
            TheNumliterals1_2_3_4_Red3 = TheNewNumliterals1_2_3_4 + TheNumRedliterals3
            TheNumliterals1_2_3_4_Red3_Red2 = TheNumliterals1_2_3_4_Red3 + TheNumRedliterals2
            TheNumliterals1_2_3_4_Red3_Red2_MAX2SAT = TheNumliterals1_2_3_4_Red3_Red2 + TheNumliteralsMAX2SAT
            TheTolTraNumliterals = TheNumliterals1_2_3_4_Red3_Red2_MAX2SAT
            # 0.5 Calculate the number of used clauses
            TheNumTotClause1_2 = TheNumClause1 + TheNumClause2
            TheNumTotClause1_2_3 = TheNumTotClause1_2 + TheNumClause3
            TheNumTotClause1_2_3_4 = TheNumTotClause1_2_3 + TheNumClause4
            TheNumTotClause1_2_3_4_Red3 = TheNumTotClause1_2_3_4 + TheNumRedClause3
            TheNumTotClause1_2_3_4_Red3_Red2 = TheNumTotClause1_2_3_4_Red3 + TheNumRedClause2
            TheNumTotClause_MAX2SAT = TheNumTotClause1_2_3_4_Red3_Red2 + TheNumClauseMAX2SAT
            # TheNumTotClause1_2_3_4_Red3_Red2 = TheNumTotClause
            # 1.1 Save the information about the number of clauses and literal
            # save the number of clause
            TheNumClauseList.append([TheNumClause1, TheNumClause2, TheNumClause3, TheNumClause4, TheNumRedClause3, TheNumRedClause2, TheNumClauseMAX2SAT, TheNumTotClause_MAX2SAT])
            # save the number of literals
            TheNumLiteralList.append([TheNumliterals1, TheNumliterals2, TheNumliterals3, TheNumliterals4, TheNumRedliterals3, TheNumRedliterals2, TheNumliteralsMAX2SAT])
            TheNumClauseList_Used.append([TheNumClause1, TheNumTotClause1_2, TheNumTotClause1_2_3, TheNumTotClause1_2_3_4, TheNumTotClause1_2_3_4_Red3, TheNumTotClause1_2_3_4_Red3_Red2, TheNumTotClause_MAX2SAT])
            TheNumLiteralList_Used.append([TheNumliterals1, TheNumliterals1_2, TheNumliterals1_2_3, TheNumliterals1_2_3_4,
                                           TheNewNumliterals1_2_3_4, TheNumliterals1_2_3_4_Red3, TheNumliterals1_2_3_4_Red3_Red2, TheTolTraNumliterals, TheTolTraNumliterals - (TheNumClause4 * 6)])
            # 1.2 Randomly generate the states of neurons and positive/negative literals
            TheNeuronState = np.zeros((TheTolTraNumliterals, NeuronString), dtype=int)  # For a CNF, the randomly assigned neuron states 1 or -1
            TheLogicRule = np.zeros(TheNumTotClause_MAX2SAT, dtype=int)  # Save all logic rules
            generate_logical_rule(TheNumClause1, TheNumTotClause1_2, TheNumTotClause1_2_3, TheNumTotClause1_2_3_4,
                                  TheNumTotClause1_2_3_4_Red3,
                                  TheNumTotClause1_2_3_4_Red3_Red2, TheNumTotClause_MAX2SAT,
                                  TheLogicRule)  # Generate random logic
            TheLogicRule_list.append(TheLogicRule)
            # print('LogicRule:', TheLogicRule)
            # Generate neuron states
            generate_learn_neuron_state(TheNumliterals1, TheNumliterals1_2, TheNumliterals1_2_3,
                                        TheNewNumliterals1_2_3_4, TheNumliterals1_2_3_4_Red3,
                                        TheNumliterals1_2_3_4_Red3_Red2,
                                        TheTolTraNumliterals, TheNumTotClause1_2_3, TheNumTotClause1_2_3_4,
                                        NeuronString, TheNeuronState, TheLogicRule)
            # 1.3 calculate Fitness
            TheFitnessNum = []  # the fitness score of the i-th logical rule
            for l in range(NeuronString):
                (FitnessNum, RatioComponent, RatioInitial, RatioInitialSatisfied,
                 RatioRed, RatioRedSatisfied, RatioTse, RatioTseSatisfied, RatioAfter, RatioAfterSatisfied) = calculate_fitness(l, TheNumClause1, TheNumTotClause1_2, TheNumTotClause1_2_3, TheNumTotClause1_2_3_4,
                                                                                                                                TheNumTotClause1_2_3_4_Red3, TheNumTotClause1_2_3_4_Red3_Red2, TheNumTotClause_MAX2SAT,
                                                                                                                                TheNumliterals1_2, TheNumliterals1_2_3, TheNewNumliterals1_2_3_4, TheNumliterals1_2_3_4_Red3,
                                                                                                                                TheNumliterals1_2_3_4_Red3_Red2, TheLogicRule, TheNeuronState)
                TheFitnessNum.append(FitnessNum)
            # 1.4 calculate errors
            # Record the first combination neuron state that meets the maximum fitness
            TheResult = np.zeros(TheTolTraNumliterals, dtype=int)
            max_point, index_max = print_point_max(TheTolTraNumliterals, TheFitnessNum, TheResult, TheNeuronState)
            calculate_Error(index_max, TheFitnessNum, TheNumTotClause_MAX2SAT, MAE, RMSE, MAPE)
            # 1.5 save synaptic weights
            DHNN(The_weight[j], max_point, TheResult, TheNumClause1, TheNumTotClause1_2, TheNumTotClause1_2_3,
                 TheNumTotClause1_2_3_4, TheNumTotClause1_2_3_4_Red3, TheNumTotClause1_2_3_4_Red3_Red2,
                 TheNumTotClause_MAX2SAT, TheNumliterals1, TheNumliterals1_2, TheNumliterals1_2_3,
                 TheNewNumliterals1_2_3_4,
                 TheNumliterals1_2_3_4_Red3, TheNumliterals1_2_3_4_Red3_Red2, TheLogicRule)
            # print(The_weight[j])
            if len(The_weight[j]) != TheNumTotClause_MAX2SAT:
                print(len(The_weight[j]))
            ## Test_list.append(max_point - TheNumTotClause)
        ## print(min(Test_list))
        # 1.6 save Minimum_energy and LogicRules
        All_LogicRules.append(TheLogicRule_list)
        Minimum_energy_list.append(Minimum_energy)
        # 1.7 Save weights
        # print('len(The_weight):', len(The_weight), 'Synaptic Weights:', The_weight)
        All_weights.append(The_weight)
        # 1.8 Save Errors
        MAE_list.append(np.mean(MAE))
        RMSE_list.append(np.mean(RMSE))
        MAPE_list.append(np.mean(MAPE))
        Standard_MAE_Learn.append(np.std(MAE))
        Standard_RMSE_Learn.append(np.std(RMSE))
        Standard_MAPE_Learn.append(np.std(MAPE))
        # 1.9 save TheInformationOfClause and Literal
        TheInformationOfClause.append(TheNumClauseList)
        TheInformationOfLiteral.append(TheNumLiteralList)
        TheInformationOfClause_Used.append(TheNumClauseList_Used)
        TheInformationOfLiteral_Used.append(TheNumLiteralList_Used)
    # print Errors
    print('MAE_Learn:', MAE_list)
    print('RMSE_Learn:', RMSE_list)
    print('MAPE_Learn:', MAPE_list)
    print('Standard_MAE_Learn:', Standard_MAE_Learn)
    print('Standard_RMSE_Learn:', Standard_RMSE_Learn)
    print('Standard_MAPE_Learn:', Standard_MAPE_Learn)
    print('-----------------------------------------------------------------------------------')
    print('\n')
    ## print('len(All_weights):', len(All_weights), '\n', 'All_weights:', All_weights)
    ## print('\n')
    #  2 The Main Function of Retrieval Phase
    # 2.0 Calculate the minimum energy
    circle = len(Minimum_energy_list)
    # print('len(Minimum_energy_list):', circle)
    # print('Minimum_energy_list:', Minimum_energy_list)
    Tol = 0.001  # Tolerance Value
    valid_states4 = {
        0: (-1, -1),
        1: (-1, -1),
        2: (-1, -1),
        3: (-1, -1),
        4: (-1, 1),
        5: (-1, 1),
        6: (-1, 1),
        7: (-1, 1),
        8: (1, -1),
        9: (1, -1),
        10: (1, -1),
        11: (1, -1),
        12: (1, 1),
        13: (1, 1),
        14: (1, 1),
        15: (1, 1),
    }
    SAT_list = []
    SAT_ratio_list = []
    Energy_list = []
    Similarity_list = []
    Similarity_list_ablation = []
    for p in range(circle):
        SAT_set, SAT_ratio_set, Energy_set, Similarity_set, Similarity_set_ablation = calculate_neurons(3, TheInformationOfClause_Used[p],
                                                                                                        TheInformationOfLiteral_Used[p], MaxComb, NeuronString,
                                                                                                        valid_states4, All_weights[p], NumTrail, Minimum_energy_list[p], Tol, All_LogicRules[p])
        SAT_list.append(SAT_set)
        SAT_ratio_list.append(SAT_ratio_set)
        Energy_list.append(Energy_set)
        Similarity_list.append(Similarity_set)
        Similarity_list_ablation.append(Similarity_set_ablation)

    print('SAT_list---->[np.mean(Z_SATSolver_index_list), np.std(Z_SATSolver_index_list), np.mean(Num_Solved_SAT), np.std(Num_Solved_SAT), Solved_SAT_Count / (Maxcomb * NumTrail)]')
    print(
        'SAT_ratio_list---->[np.mean(Ratio_Solved_Component), np.mean(Ratio_Solved_Initial), np.mean(Ratio_Solved_Initial_Satisfied), np.mean(Ratio_Solved_Red), np.mean(Ratio_Solved_Red_Satisfied), np.mean(Ratio_Solved_Tse), np.mean(Ratio_Solved_Tse_Satisfied), np.mean(Ratio_Solved_After), np.mean(Ratio_Solved_After_Satisfied)]')
    print('Energy_list---->[MAE_Energy, Ratio of global minimum solutions, Ratio of local minimum solutions]')

    print('Similarity_list---->[np.mean(Similarity_list_J_Initial), np.std(Similarity_list_J_Initial), np.mean(Similarity_list_H_Initial), np.std(Similarity_list_H_Initial)]')
    print(
        'Similarity_list_ablation---->[np.mean(Similarity_list_J_Red), np.std(Similarity_list_J_Red), np.mean(Similarity_list_H_Red), np.std(Similarity_list_H_Red),np.mean(Similarity_list_J_Tse), np.std(Similarity_list_J_Tse), np.mean(Similarity_list_H_Tse), np.std(Similarity_list_H_Tse), np.mean(Similarity_list_J_After), np.std(Similarity_list_J_After), np.mean(Similarity_list_H_After), np.std(Similarity_list_H_After)]')
    print(len(SAT_list), 'SAT_list:', SAT_list)
    print(len(SAT_ratio_list), 'SAT_ratio_list:', SAT_ratio_list)
    print(len(Energy_list), 'Energy_list:', Energy_list)
    print(len(Similarity_list), 'Similarity_list:', Similarity_list)
    print(len(Similarity_list_ablation), 'Similarity_list_ablation:', Similarity_list_ablation)

    # Define the first column as N
    # ===============================
    N = np.arange(6, 6 * 21, 6)  # Starting from 6, with a step size of 6, a total of 20 numbers.
    df_front = pd.DataFrame({
        "N": N,
        "MAE_Learn": MAE_list,
        "RMSE_Learn": RMSE_list,
        "MAPE_Learn": MAPE_list,
        "Standard_MAE_Learn": Standard_MAE_Learn,
        "Standard_RMSE_Learn": Standard_RMSE_Learn,
        "Standard_MAPE_Learn": Standard_MAPE_Learn
    })
    # ===============================
    # Define the table headers
    # ===============================
    SAT_header = ['np.mean(Z_SATSolver_index_list)', 'np.std(Z_SATSolver_index_list)', 'np.mean(Num_Solved_SAT)', 'np.std(Num_Solved_SAT)', 'Solved_SAT_Count / (Maxcomb * NumTrail)']
    SAT_ratio_header = ['np.mean(Ratio_Solved_Component)', 'np.mean(Ratio_Solved_Initial)', 'np.mean(Ratio_Solved_Initial_Satisfied)', 'np.mean(Ratio_Solved_Red)',
                        'np.mean(Ratio_Solved_Red_Satisfied)', 'np.mean(Ratio_Solved_Tse)', 'np.mean(Ratio_Solved_Tse_Satisfied)', 'np.mean(Ratio_Solved_After)', 'np.mean(Ratio_Solved_After_Satisfied)']
    Energy_header = ["MAE_Energy", "Ratio of global minimum solutions", "Ratio of local minimum solutions"]
    Similarity_header = ["np.mean(Similarity_list_J_Initial)", "np.std(Similarity_list_J_Initial)", "np.mean(Similarity_list_H_Initial)", "np.std(Similarity_list_H_Initial)"]
    Similarity_ablation_header = ['np.mean(Similarity_list_J_Red)', 'np.std(Similarity_list_J_Red)', 'np.mean(Similarity_list_H_Red)', 'np.std(Similarity_list_H_Red)',
                                  'np.mean(Similarity_list_J_Tse)', 'np.std(Similarity_list_J_Tse)', 'np.mean(Similarity_list_H_Tse)', 'np.std(Similarity_list_H_Tse)',
                                  'np.mean(Similarity_list_J_After)', 'np.std(Similarity_list_J_After)', 'np.mean(Similarity_list_H_After)', 'np.std(Similarity_list_H_After)']
    # ---- Convert to DataFrame ----
    df_SAT = pd.DataFrame(SAT_list, columns=SAT_header)
    df_ratio_SAT = pd.DataFrame(SAT_ratio_list, columns=SAT_ratio_header)
    df_Energy = pd.DataFrame(Energy_list, columns=Energy_header)
    df_Similarity = pd.DataFrame(Similarity_list, columns=Similarity_header)
    df_Similarity_ablation = pd.DataFrame(Similarity_list_ablation, columns=Similarity_ablation_header)

    df_all = pd.concat([df_front, df_SAT, df_ratio_SAT, df_Energy, df_Similarity, df_Similarity_ablation], axis=1)

    # ===============================
    # Save to Excel
    # ===============================
    df_all.to_excel("ZRAN2,3SAT.xlsx", sheet_name="AllData", index=False)

    print("The data has been successfully exported to .xlsx")


if __name__ == "__main__":
    main()

end = time.perf_counter()

print(f"The running time: {end - start:.6f} second")
