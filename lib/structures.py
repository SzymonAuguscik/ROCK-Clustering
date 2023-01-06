from functools import reduce
from heapq import heapify, heappush


class Point:
    id = 0
    def __init__(self, x, y, true_class):
        self.x = x
        self.y = y
        self.true_class = true_class
        self.predicted_class = -1
        self.id = Point.id
        Point.id += 1

    def __str__(self):
        return f"Point({self.id})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, Point) and self.id == other.id

    def __hash__(self):
        return hash(self.id)
    
    @staticmethod
    def distance(point):
        return point.x**2 + point.y**2

class Cluster:
    id = 0
    
    def __init__(self, points):
        self.points = points
        self.id = Cluster.id
        Cluster.id += 1

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Cluster) and self.id == other.id

    def __str__(self):
        return f"Cluster({self.id})\n" + reduce(lambda p1, p2: str(p1) + " " + str(p2), self.points, "") + '\n'

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return self.id < other.id

class LocalHeap:
    def __init__(self, cluster, linked_clusters, goodness_measure):
        self.cluster = cluster
        self.goodness_measure = goodness_measure
        self.heap = [(self.goodness_measure(c, self.cluster), c) for c in linked_clusters]
        heapify(self.heap)

    def get_linked_clusters(self):
        return [c for (_, c) in self.heap]

    def get_max_linked_cluster(self):
        return self.heap[-1][1] if self.heap else None

    def delete(self, cluster):
        self.heap[:] = (h for h in self.heap if h[1] != cluster)

    def insert(self, cluster):
        heappush(self.heap, (self.goodness_measure(cluster, self.cluster), cluster))

class GlobalHeap:
    def __init__(self, clusters):
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def get_max(self):
        return self.clusters[-1]

    def delete(self, cluster):
        self.clusters[:] = (c for c in self.clusters if c != cluster)

    def update(self, local_heaps, key):
        self.clusters = list(map(lambda heap: heap.cluster, sorted(local_heaps, key=key)))

    def insert(self, cluster, local_heaps, key):
        self.clusters.append(cluster)
        self.update(local_heaps, key)

