import numpy as np

from math import sqrt
from itertools import combinations


def link_points(points):
    return list(combinations(points, 2))

def neighbour_estimation_function_1(theta):
    return (1 + theta) / (1 - theta)

def neighbour_estimation_function_2(theta):
    return np.exp(theta) / (1 - theta)

def neighbour_estimation_function_3(theta):
    return np.sin(theta) / (1 - theta)

def neighbour_estimation_function_4(theta):
    return 50 * theta + 1

def neighbour_estimation_function_5(theta):
    return (2 - theta) / (1 + theta)
    
def max_l1_distance(data):
    return (abs(max(data.iloc[:, 0]) - min(data.iloc[:, 0])) + abs(max(data.iloc[:, 1]) - min(data.iloc[:, 1])))
    
def max_l2_distance(data):
    return sqrt((max(data.iloc[:, 0]) - min(data.iloc[:, 0]))**2 + (max(data.iloc[:, 1]) - min(data.iloc[:, 1]))**2)

def l1_metric(p1, p2):
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)

def l2_metric(p1, p2):
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def closeness_l1(p1, p2, max_distance):
    return (max_distance - l2_metric(p1, p2)) / max_distance

def closeness_l2(p1, p2, max_distance):
    return (max_distance - l1_metric(p1, p2)) / max_distance

def neighbours_count(point, neighbours):
    return len(neighbours[point])