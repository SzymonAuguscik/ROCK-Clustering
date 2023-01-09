import os

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score, silhouette_samples, rand_score
from matplotlib import colors as mcolors
from pyclustering.cluster.rock import rock
from lib.RockClustering import RockClustering
from functools import reduce
from time import time

if __name__ == "__main__":
    gdf = gpd.read_file('datasets/onlinedeliverydata.csv')

    gdf['Monthly Income'] = pd.factorize(gdf['Monthly Income'])[0]
    gdf[['longitude', 'latitude']] = gdf[['longitude', 'latitude']].apply(pd.to_numeric)

    gdf = gdf[['longitude', 'latitude', 'Monthly Income']]

    my_rock = RockClustering(gdf, len(np.unique(gdf['Monthly Income'])), __file__, 0.8, neighbours='n5', dataset_part=0.4)
    my_rock.perform_clustering()

    points = [[row['longitude'], row['latitude']] for _, row in gdf.iterrows()]

    t1 = time()
    rock_instance = rock(points, 0.045, len(np.unique(gdf['Monthly Income'])))
    rock_instance.process()
    clusters = rock_instance.get_clusters()
    t2 = time() - t1

    scores = { "time" : round(t2, 3) }
    true_classes = [x for x in gdf['Monthly Income']]
    predicted_classes = [x.index(True) if True in x else -1 for x in [[p in c for c in clusters] for p in range(len(points))]]
    
    samples_silhouette_score = silhouette_samples(gdf.iloc[:, :-1], predicted_classes)
    rand_index = rand_score(true_classes, predicted_classes)

    scores["clustered_points_number"] = reduce(lambda num, c: num + len(c), clusters, 0)
    scores["clustering_silhouette_score"] = round(silhouette_score(gdf.iloc[:, :-1], predicted_classes), 3)
    scores["samples_silhouette_min_score"] = round(min(samples_silhouette_score), 3)
    scores["samples_silhouette_max_score"] = round(max(samples_silhouette_score), 3)
    scores["rand_index"] = round(rand_index, 3)
    
    filename = os.path.join("scores", os.path.basename(__file__).split('.')[0])
    with open(f"{os.path.join(os.path.dirname(__file__), filename)}_pyclustering.json", 'w') as file:
        file.write(str(scores))

    plt.clf()
    colors = mcolors.TABLEAU_COLORS

    for cluster, color in zip(clusters, colors):
        plt.scatter([points[idx][0] for idx in cluster], [points[idx][1] for idx in cluster], c=color)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Clustered samples - pyclustering ROCK")
    filename = os.path.join("images", os.path.basename(__file__).split('.')[0])
    plt.savefig(f"{os.path.join(os.path.dirname(__file__), filename)}_pyclustering")

