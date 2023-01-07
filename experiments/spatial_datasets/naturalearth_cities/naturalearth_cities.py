import geopandas as gpd
import pandas as pd
import numpy as np

from lib.RockClustering import RockClustering

if __name__ == "__main__":
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
    continents = world[['geometry', 'continent']]

    gdf = cities.sjoin(continents, how="inner", predicate='intersects')

    gdf['continent'] = pd.factorize(gdf['continent'])[0]
    gdf['x'] = gdf['geometry'].apply(lambda p: p.x)
    gdf['y'] = gdf['geometry'].apply(lambda p: p.y)

    gdf = gdf[['x', 'y', 'continent']]

    rock = RockClustering(gdf, len(np.unique(gdf['continent'])), __file__, 0.85)
    rock.perform_clustering()