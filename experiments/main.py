import numpy as np
from scipy.io import arff
from lib.RockClustering import RockClustering

# export PYTHONPATH="${PYTHONPATH}:/home/szymon/Pulpit/SPDB"

if __name__ == "__main__":
    sample = 1000

    dataset = arff.loadarff('datasets/2d-4c-no4.arff')
    x = [d[0] for d in dataset[0][:sample]]
    y = [d[1] for d in dataset[0][:sample]]
    print(np.array(dataset))

    # rock = RockClustering((x,y), 4, 0.7)
    rock = RockClustering((x,y), 4, 0.9)
    rock.perform_clustering()

