import pandas as pd

from scipy.io import arff
from lib.RockClustering import RockClustering

if __name__ == "__main__":
    dataset = arff.loadarff('datasets/2d-4c-no4.arff')
    df = pd.DataFrame(dataset[0])
    df['class'] = pd.factorize(df['class'])[0]

    rock = RockClustering(df, 4, __file__, 0.8)
    rock.perform_clustering()

