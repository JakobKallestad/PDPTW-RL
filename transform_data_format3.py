from generate_pdp_data import load_dataset
import numpy as np
import os
import pickle


CAPACITY_MAP = {
    10: 10,  # 20,
    20: 15,  # 30,
    50: 20,  # 40,
    100: 25,  # 50
}


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(filename, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


# euclidean distance
def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


class PDP:
    def __init__(self, size, n_calls, locations, capacities, calls, dist_matrix):
        self.size = size
        self.n_calls = n_calls
        self.locations = locations
        self.capacities = capacities
        self.calls = calls
        self.dist_matrix = dist_matrix

    def save_problem(self):
        pass


def transform_dataset(dataset):
    new_dataset = []
    for i, data in enumerate(dataset, 1):
        new_dataset.append(transform_data(data))
        print(i)
    return new_dataset


def transform_data(data):

    # data information
    size = len(data['loc'])
    n_calls = size // 2
    locations = [data['depot'].tolist()] + data['loc'].tolist()
    locations = np.asarray(locations)
    capacities = data['demand']#.tolist()
    calls = [(None, None)] + [(i, i + 1) for i in range(1, size, 2)]
    dist_matrix = np.empty((size + 1, size + 1), dtype=np.float)
    for i in range(size+1):
        for j in range(i, size+1):
            d = distance(locations[i], locations[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    pdp = PDP(size, n_calls, locations, capacities, calls, dist_matrix)
    return pdp

GRAPH_SIZE = 20
NAME = "TEST1"
SEED = 1234

dataset = load_dataset("data/pdp/pdp20_TEST1_seed1234.pkl")
new_dataset = transform_dataset(dataset)
save_dataset(new_dataset, f'transformed_data3/pdp{GRAPH_SIZE}_{NAME}_seed{SEED}.pkl')


