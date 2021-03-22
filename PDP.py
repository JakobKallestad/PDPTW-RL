import random

import numpy as np
from Utils import get_distance
import pickle


CAPACITY_MAP = {
    10: 10,  # 20,
    20: 15,  # 30,
    50: 20,  # 40,
    100: 25,  # 50
}


class PDP:
    def __init__(self, size, n_calls, locations, capacities, calls, dist_matrix):
        self.size = size
        self.n_calls = n_calls
        self.locations = locations
        self.capacities = capacities
        self.calls = calls
        self.dist_matrix = dist_matrix
        #self.initialize_close_calls()

    def save_problem(self):
        with open('filename.pkl', 'wb') as file:
            pickle.dump([self], file, protocol=pickle.HIGHEST_PROTOCOL)

    def initialize_close_calls(self):
        self.distances = self.calculate_close_calls()

    def calculate_close_calls(self):
        max_sim_size = 11
        distances = [None] * max_sim_size
        prev_dists = list(zip([frozenset([i]) for i in range(1, self.n_calls + 1)], [0] * self.n_calls))
        for q in range(2, max_sim_size):
            new_dists = self.calc_new_dists(prev_dists)
            new_dists = sorted(new_dists, key=lambda x: x[1])  # memory limiting factor
            distances[q] = new_dists[:200]
            prev_dists = new_dists[:1000]  # beam width
        return distances

    def calc_new_dists(self, prev_dists):
        new_dists = []
        memory = set()
        for indexes, dist in prev_dists:
            for i in set(range(1, self.n_calls + 1)) - indexes:
                expanded_indexes = frozenset([*indexes, i])
                if expanded_indexes in memory:
                    continue
                memory.add(expanded_indexes)
                total = dist
                for ind in indexes:
                    total += (self.dist_matrix[self.calls[i][0], self.calls[ind][0]] +
                              self.dist_matrix[self.calls[i][1], self.calls[ind][1]])
                new_dists.append([expanded_indexes, total])
        return new_dists


def generate_problem(size=20):
    locations = np.random.uniform(size=(size+1, 2))  # location 0 is depot
    capacities = np.random.randint(1, 10, size=(size // 2)).repeat(2) / CAPACITY_MAP.get(size)
    capacities[1::2] *= -1

    n_calls = size // 2
    calls = [(None, None)] + [(i, i + 1) for i in range(1, size, 2)]
    dist_matrix = np.empty((size+1, size+1), dtype=np.float)
    for i in range(size+1):
        for j in range(i, size+1):
            d = get_distance(locations[i], locations[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    pdp = PDP(size, n_calls, locations, capacities, calls, dist_matrix)
    return pdp


def load_pdp_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_dataset(dataset=[], path='data/TEST.pkl'):
    with open(path, 'wb') as file:
        pickle.dump(dataset, file, protocol=pickle.HIGHEST_PROTOCOL)


def generate_dataset_to_file(n_instances=10, SIZE=20, SEED=1234):
    np.random.seed(SEED)
    random.seed(SEED)
    dataset = []
    for i in range(n_instances):
        print(i, "/", n_instances)
        data = generate_problem(size=SIZE)
        dataset.append(data)
    save_dataset(dataset, path=f'data/pdp_{SIZE}/seed{SEED}_size{SIZE}_num{n_instances}.pkl')


#n_instances = 100
#SIZE = 100
#SEED = 1234
#generate_dataset_to_file(n_instances, SIZE, SEED)
