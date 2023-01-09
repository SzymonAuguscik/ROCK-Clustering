import os
import random

from . import utils
from time import time
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from .structures import Point, Cluster, LocalHeap, GlobalHeap 
from sklearn.metrics import silhouette_score, silhouette_samples, rand_score


class RockClustering:
    def __init__(self,
                 data,
                 K,
                 filename,
                 theta=0.5,
                 neighbours='n1',
                 distance='l2',
                 dataset_part=1):
        self.data = data
        self.K = K
        self._set_up_file_paths(filename)
        self.theta = theta
        self.neighbour_estimation_function = {
            'n1' : utils.neighbour_estimation_function_1,
            'n2' : utils.neighbour_estimation_function_2,
            'n3' : utils.neighbour_estimation_function_3,
            'n4' : utils.neighbour_estimation_function_4,
            'n5' : utils.neighbour_estimation_function_5
        }[neighbours]
        self.similarity_function = utils.closeness_l1 if distance == 'l1' else utils.closeness_l2
        self.max_distance_function = utils.max_l1_distance if distance == 'l1' else utils.max_l2_distance
        self.scores = {}
        self.dataset_part = dataset_part if dataset_part != 1 else None
        self.points_under_clustering = None
        self.remaining_points = None
        self.outliers = None
        self.neighbours = None
        self.links = None
        self.clusters = None
        self.local_heaps = None
        self.global_heap = None

    def _set_up_file_paths(self, filename):
        self.images_dir = os.path.join(os.path.dirname(filename), "images")
        self.scores_dir = os.path.join(os.path.dirname(filename), "scores")
        self.filename = os.path.splitext(os.path.basename(filename))[0]

        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)

        if not os.path.exists(self.scores_dir):
            os.makedirs(self.scores_dir)

    def _get_points(self):
        x = self.data.iloc[:, 0].to_list()
        y = self.data.iloc[:, 1].to_list()
        classes = self.data.iloc[:, 2].to_list()
        return [Point(i, j, c) for i, j, c in zip(x, y, classes)]

    def _get_neighbours(self):
        max_distance = self.max_distance_function(self.data)
        return {p1 : set([p2 for p2 in self.points_under_clustering if p2 != p1 and self.similarity_function(p1, p2, max_distance) >= self.theta]) for p1 in self.points_under_clustering}

    def _extract_outliers(self):
        outliers = [point for point in self.points_under_clustering if len(self.neighbours[point]) == 0]
        self.points_under_clustering = [point for point in self.points_under_clustering if len(self.neighbours[point]) > 0]
        return outliers

    def _extract_remaining_points(self):
        k = int(len(self.points_under_clustering) * (1 - self.dataset_part))
        remaining_points = random.sample(self.points_under_clustering, k)
        
        for point in remaining_points:
            self.points_under_clustering.remove(point)

        return remaining_points

    def _get_points_link(self, point1, point2):
        if point1 == point2:
            return 0

        neighbours1 = self.neighbours[point1]
        neighbours2 = self.neighbours[point2]

        return len(neighbours1 & neighbours2)

    def _compute_links(self):
        links = {}

        for i, point1 in enumerate(self.points_under_clustering):
            for point2 in self.points_under_clustering[i + 1:]:
                links[tuple(sorted([point1.id, point2.id]))] = self._get_points_link(point1, point2)

        return links

    def _get_initial_clusters(self):
        return [Cluster([point]) for point in self.points_under_clustering]

    def _get_clusters_link(self, cluster1, cluster2):
        return sum([self._get_points_link(p1, p2) for p1 in cluster1.points for p2 in cluster2.points])

    def _get_goodness_measure(self, cluster1, cluster2):
        if not (cluster1 and cluster2):
            return 0

        n1, n2 = len(cluster1.points), len(cluster2.points)
        power = 1 + 2 * self.neighbour_estimation_function(self.theta)
        return self._get_clusters_link(cluster1, cluster2) / ((n1 + n2)**power - n1**power - n2**power)
        
    def _create_local_heap(self, cluster):
        heap = [c for c in self.clusters if self._get_clusters_link(cluster, c) > 0 and c != cluster]
        return LocalHeap(cluster, heap, self._get_goodness_measure)

    def _create_local_heaps(self):
        return [self._create_local_heap(cluster) for cluster in self.clusters]

    def _create_global_heap(self):
        clusters = list(map(lambda heap: heap.cluster,
                   sorted(self.local_heaps,
                          key=lambda heap: self._get_goodness_measure(heap.cluster, heap.get_max_linked_cluster()))))
        return GlobalHeap(clusters)

    def _get_local_heap_for_cluster(self, cluster):
        return list(filter(lambda heap: cluster == heap.cluster, self.local_heaps))[0]

    def _delete_cluster(self, cluster):
        self.Q.delete(cluster)
        self.local_heaps[:] = (heap for heap in self.local_heaps if heap.cluster != cluster)
        self.clusters[:] = (c for c in self.clusters if c != cluster)

    def _get_true_classes(self):
        return [point.true_class for point in sorted(self.points_under_clustering + self.outliers,
                                                     key=lambda p: p.id)]

    def _get_predicted_classes(self):
        return [point.predicted_class for point in sorted(self.points_under_clustering + self.outliers,
                                                          key=lambda p: p.id)]

    def _get_partial_goodness_measure(self, neighbourins_count, cluster_count):
        return neighbourins_count / (cluster_count + 1)**self.neighbour_estimation_function(self.theta)

    def _cluster_remaining_points(self):
        self.remaining_points = sorted(self.remaining_points,
                                       reverse=False,
                                       key=lambda point: len(set(self.neighbours[point]) & set(self.points_under_clustering)))

        for point in self.remaining_points:
            print(f"{point} : {len(set(self.neighbours[point]) & set(self.points_under_clustering))}")

        while len(self.remaining_points):
            print(len(self.remaining_points))
            point = self.remaining_points.pop()
            point_neighbours = self.neighbours[point]
            self.points_under_clustering.append(point)
            
            if len(set(point_neighbours) & set(self.points_under_clustering)) == 0:
                continue
            
            neighbouring_clusters = { cluster : neighbours for cluster, neighbours in
                                    { c : [point for point in c.points if point in point_neighbours] for c in self.Q.clusters }.items()
                                      if len(neighbours)}

            if not neighbouring_clusters:
                continue

            best_cluster = max(neighbouring_clusters,
                               key=lambda c: self._get_partial_goodness_measure(len(point_neighbours),
                                                                                len(neighbouring_clusters[c])))
            best_cluster.points.append(point)

        self.points_under_clustering = sorted(self.points_under_clustering, key=lambda p: p.id)

    def perform_clustering(self):
        t1 = time()
        print("Start!")
        print(f"Clusters to be set: {self.K}")
        self.points_under_clustering = self._get_points()
        print("Points")
        self.neighbours = self._get_neighbours()
        self.outliers = self._extract_outliers()
        print("Neighbours")

        if self.dataset_part:
            self.remaining_points = self._extract_remaining_points()

        self.links = self._compute_links()
        print("Links")
        self.clusters = self._get_initial_clusters()
        print("Clusters")
        self.local_heaps = self._create_local_heaps()
        print("Local heaps")
        self.Q = self._create_global_heap()
        print("Q")

        self.draw_raw_points()

        while len(self.Q) > self.K:
            print(len(self.Q))
            u = self.Q.get_max()
            qu = self._get_local_heap_for_cluster(u)
            v = qu.get_max_linked_cluster()

            if not v:
                self._delete_cluster(u)
                continue

            w = Cluster(list(set(u.points) | set(v.points)))
            self.clusters.append(w)

            qw = LocalHeap(w, [], self._get_goodness_measure)
            qv = self._get_local_heap_for_cluster(v)

            self._delete_cluster(u)
            self._delete_cluster(v)

            update_clusters = lambda local_heap: self._get_goodness_measure(local_heap.cluster, local_heap.get_max_linked_cluster())

            for x in (set(qu.get_linked_clusters()) | set(qv.get_linked_clusters())) - set((u, v)):
                u_link = self.links[tuple(sorted([x.id, u.id]))] if tuple(sorted([x.id, u.id])) in self.links else 0
                v_link = self.links[tuple(sorted([x.id, v.id]))] if tuple(sorted([x.id, v.id])) in self.links else 0
                self.links[tuple(sorted([x.id, w.id]))] = u_link + v_link
                qx = self._get_local_heap_for_cluster(x)
                qx.delete(u)
                qx.delete(v)
                qx.insert(w)
                qw.insert(x)
                self.Q.update(self.local_heaps, update_clusters)

            self.local_heaps.append(qw)
            self.Q.insert(w, self.local_heaps, update_clusters)

        if self.dataset_part:
            self._cluster_remaining_points()

        for i, cluster in enumerate(self.Q.clusters):
            for point in cluster.points:
                point.predicted_class = i

        t2 = time() - t1
        self.scores["time_in_seconds"] = round(t2, 3)
        self._save_scores()
        self.draw_clustered_points()
        print("End!")

    def _save_scores(self):
        true_classes = self._get_true_classes()
        predicted_classes = self._get_predicted_classes()

        samples_silhouette_score = silhouette_samples(self.data.iloc[:, :-1], predicted_classes)
        rand_index = rand_score(true_classes, predicted_classes)

        self.scores["clustered_points_number"] = sum([len(c.points) for c in self.Q.clusters])
        self.scores["clustering_silhouette_score"] = round(silhouette_score(self.data.iloc[:, :-1], predicted_classes), 3)
        self.scores["samples_silhouette_min_score"] = round(min(samples_silhouette_score), 3)
        self.scores["samples_silhouette_max_score"] = round(max(samples_silhouette_score), 3)
        self.scores["rand_index"] = round(rand_index, 3)
        
        with open(f"{os.path.join(self.scores_dir, self.filename)}.json", 'w') as file:
           file.write(str(self.scores))

    def draw_raw_points(self):
        plt.clf()
        plt.scatter(self.data.iloc[:, 0].to_list(), self.data.iloc[:, 1].to_list())
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Samples")
        plt.savefig(f"{os.path.join(self.images_dir, self.filename)}_samples")

        for source, neighbours in self.neighbours.items():
            for neighbour in neighbours:
                plt.plot((source.x, neighbour.x), (source.y, neighbour.y), 'r')

        plt.title("Neighbours")
        plt.savefig(f"{os.path.join(self.images_dir, self.filename)}_neighbours")

    def draw_clustered_points(self):
        plt.clf()
        colors = mcolors.TABLEAU_COLORS

        for cluster, color in zip(self.Q.clusters, colors):
            plt.scatter([point.x for point in cluster.points], [point.y for point in cluster.points], c=color)

        plt.scatter([outlier.x for outlier in self.outliers], [outlier.y for outlier in self.outliers], c='k')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Clustered samples")
        plt.savefig(f"{os.path.join(self.images_dir, self.filename)}_clusters")

