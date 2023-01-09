import geopandas as gpd
import pandas as pd

from lib.RockClustering import RockClustering

if __name__ == "__main__":
    gdf = gpd.read_file('datasets/database.csv')
    gdf.drop(gdf[~gdf['Date'].str.contains('2016|2015')].index, inplace=True)

    gdf[['Longitude', 'Latitude', 'Magnitude']] = gdf[['Longitude', 'Latitude', 'Magnitude']].apply(pd.to_numeric)
    classes = 5
    gdf['Magnitude'] = pd.qcut(gdf['Magnitude'], q=classes)
    gdf['Magnitude'] = pd.factorize(gdf['Magnitude'])[0]

    gdf = gdf[['Longitude', 'Latitude', 'Magnitude']]

    rock = RockClustering(gdf, classes, __file__, 0.9)
    rock.perform_clustering()