from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from .structures import Point, Cluster, LocalHeap, GlobalHeap 
from .utils import closeness_l1, max_l1_distance, neighbour_estimation_function, sort


class RockClustering:
    def __init__(self, data, K, theta=0.5):
        self.data = data
        self.K = K
        self.theta = theta
        self.points = None
        self.outliers = None
        self.neighbours = None
        self.links = None
        self.clusters = None
        self.local_heaps = None
        self.global_heap = None

    def _get_points(self):
        x, y = self.data
        return [Point(i, j) for i, j in zip(x, y)]

    def _get_neighbours(self, sim, max_distance_func):
        max_distance = max_distance_func(self.data)
        return {p1 : set([p2 for p2 in self.points if p2 != p1 and sim(p1, p2, max_distance) >= self.theta]) for p1 in self.points}

    def _extract_outliers(self):
        outliers = [point for point in self.points if len(self.neighbours[point]) == 0]
        self.points = [point for point in self.points if len(self.neighbours[point]) > 0]
        return outliers

    def _get_points_link(self, point1, point2):
        if point1 == point2:
            return 0

        neighbours1 = self.neighbours[point1]
        neighbours2 = self.neighbours[point2]

        return len(neighbours1 & neighbours2)

    def _compute_links(self):
        links = {}

        for i, point1 in enumerate(self.points):
            for point2 in self.points[i + 1:]:
                links[tuple(sorted([point1.id, point2.id]))] = self._get_points_link(point1, point2)

        return links

    def _get_initial_clusters(self):
        return [Cluster([point]) for point in self.points]

    def _get_clusters_link(self, cluster1, cluster2):
        return sum([self._get_points_link(p1, p2) for p1 in cluster1.points for p2 in cluster2.points])

    def _get_goodness_measure(self, cluster1, cluster2, f=neighbour_estimation_function):
        if not (cluster1 and cluster2):
            return 0

        n1, n2 = len(cluster1.points), len(cluster2.points)
        power = 1 + 2 * f(self.theta)
        return self._get_clusters_link(cluster1, cluster2) / ((n1 + n2)**power - n1**power - n2**power)
        
    def _create_local_heap(self, cluster):
        heap = [c for c in self.clusters if self._get_clusters_link(cluster, c) > 0 and c != cluster]
        return LocalHeap(cluster, heap, self._get_goodness_measure)

    def _create_local_heaps(self):
        return [self._create_local_heap(cluster) for cluster in self.clusters]

    def _create_global_heap(self):
        clusters = list(map(lambda heap: heap.cluster,
                   sort(self.local_heaps,
                        lambda heap: self._get_goodness_measure(heap.cluster, heap.get_max_linked_cluster()))))
        return GlobalHeap(clusters)

    def _get_local_heap_for_cluster(self, cluster):
        return list(filter(lambda heap: cluster == heap.cluster, self.local_heaps))[0]

    def _delete_cluster(self, cluster):
        self.Q.delete(cluster)
        self.local_heaps[:] = (heap for heap in self.local_heaps if heap.cluster != cluster)
        self.clusters[:] = (c for c in self.clusters if c != cluster)

    def _add_cluster(self, cluster):
        self.clusters.append(cluster)

    def load_data(self):
        pass

    def perform_clustering(self):
        from time import time
        t1 = time()
        print("Start!")
        self.points = self._get_points()
        print("self.points")
        self.neighbours = self._get_neighbours(closeness_l1, max_l1_distance)
        self.outliers = self._extract_outliers()

        for k in self.neighbours:
            print(f"{k} = {len(self.neighbours[k])}")

        print("self.neighbours")
        self.links = self._compute_links()
        print("self.links")
        self.clusters = self._get_initial_clusters()
        print("self.clusters")
        self.local_heaps = self._create_local_heaps()
        print("self.local_heaps")
        self.Q = self._create_global_heap()
        print("self.Q")

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
            self._add_cluster(w)

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

        t2 = time() - t1
        print(f"Time: {t2} s")
        print(f"Clustered points: {sum([len(c.points) for c in self.Q.clusters])}")
        self.draw_points()

    def print_results(self):
        pass

    def get_final_clusters(self):
        pass

    def draw_raw_points(self):
        plt.scatter(*self.data)

        for source, neighbours in self.neighbours.items():
            for neighbour in neighbours:
                plt.plot((source.x, neighbour.x), (source.y, neighbour.y), 'r')

        plt.show()

    def draw_points(self):
        colors = mcolors.TABLEAU_COLORS

        for cluster, color in zip(self.Q.clusters, colors):
            plt.scatter([point.x for point in cluster.points], [point.y for point in cluster.points], c=color)

        plt.scatter([outlier.x for outlier in self.outliers], [outlier.y for outlier in self.outliers], c='k')
        plt.show()

