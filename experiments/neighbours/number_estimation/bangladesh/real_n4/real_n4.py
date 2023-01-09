import geopandas as gpd
import pandas as pd
import numpy as np

from lib.RockClustering import RockClustering

if __name__ == "__main__":
    gdf = gpd.read_file('datasets/bangladesh_cleaned.csv')

    gdf['region'] = pd.factorize(gdf['region'])[0]
    gdf[['longitude', 'latitude']] = gdf[['longitude', 'latitude']].apply(pd.to_numeric)
    
    gdf = gdf[['longitude', 'latitude', 'region']]

    rock = RockClustering(gdf, len(np.unique(gdf['region'])), __file__, 0.92, neighbours='n4')
    rock.perform_clustering()

