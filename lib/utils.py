from math import sqrt
from itertools import combinations


def link_points(points):
    return list(combinations(points, 2))

def neighbour_estimation_function(theta):
    return (1 + theta) / (1 - theta)

def sort(sortable, key):
    return sorted(sortable, key=key)
    
def max_l1_distance(data):
    return sqrt((min(data[0]) - min(data[1]))**2 + (max(data[0]) - max(data[1]))**2)
    
def max_l2_distance(data):
    return (abs(min(data[0]) - min(data[1])) + abs(max(data[0]) - max(data[1])))

def l1_metric(p1, p2):
    return sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def l2_metric(p1, p2):
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)

def closeness_l1(p1, p2, max_distance):
    return (max_distance - l1_metric(p1, p2)) / max_distance

def closeness_l2(p1, p2, max_distance):
    return (max_distance - l2_metric(p1, p2)) / max_distance

def neighbours_count(point, neighbours):
    return len(neighbours[point])