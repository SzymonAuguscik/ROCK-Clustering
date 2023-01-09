import geopandas as gpd
import pandas as pd
import numpy as np

from lib.RockClustering import RockClustering

if __name__ == "__main__":
    gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    gdf['geometry'] = gdf['geometry'].apply(lambda g: g.centroid)
    gdf['x'] = gdf['geometry'].apply(lambda p: p.x)
    gdf['y'] = gdf['geometry'].apply(lambda p: p.y)
    gdf['continent'] = pd.factorize(gdf['continent'])[0]

    gdf = gdf[['x', 'y', 'continent']]

    rock = RockClustering(gdf, len(np.unique(gdf['continent'])), __file__, 0.92, distance='l2')
    rock.perform_clustering()