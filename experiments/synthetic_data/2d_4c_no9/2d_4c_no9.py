import pandas as pd
import numpy as np

from scipy.io import arff
from lib.RockClustering import RockClustering

if __name__ == "__main__":
    dataset = arff.loadarff('datasets/2d-4c-no9.arff')
    df = pd.DataFrame(dataset[0])
    df['class'] = pd.factorize(df['class'])[0]

    rock = RockClustering(df, len(np.unique(df['class'])), __file__, 0.75)
    rock.perform_clustering()

