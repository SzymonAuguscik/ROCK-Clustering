import geopandas as gpd
import pandas as pd
import numpy as np

from lib.RockClustering import RockClustering

if __name__ == "__main__":
    gdf = gpd.read_file('datasets/onlinedeliverydata.csv')

    gdf['Monthly Income'] = pd.factorize(gdf['Monthly Income'])[0]
    gdf[['longitude', 'latitude']] = gdf[['longitude', 'latitude']].apply(pd.to_numeric)
    gdf[['longitude', 'latitude']] = gdf[['longitude', 'latitude']].apply(lambda x: 1000 * (x - x.min()) / (x.max() - x.min()))

    gdf = gdf[['longitude', 'latitude', 'Monthly Income']]

    rock = RockClustering(gdf, len(np.unique(gdf['Monthly Income'])), __file__, 0.85)
    rock.perform_clustering()