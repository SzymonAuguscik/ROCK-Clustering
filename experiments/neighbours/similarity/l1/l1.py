import pandas as pd
import numpy as np

from scipy.io import arff
from lib.RockClustering import RockClustering

if __name__ == "__main__":
    dataset = arff.loadarff('datasets/square5.arff')
    df = pd.DataFrame(dataset[0])
    df['class'] = pd.factorize(df['class'])[0]

    rock = RockClustering(df, len(np.unique(df['class'])), __file__, 0.9, distance='l1')
    rock.perform_clustering()

