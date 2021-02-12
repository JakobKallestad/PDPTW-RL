import numpy as np
import random
import copy
import math

def calculate_distance(point0, point1):
    dx = point1[0] - point0[0]
    dy = point1[1] - point0[1]
    return math.sqrt(dx * dx + dy * dy)

class Problem:
    def __init__(self, locations, capacities):
        self.locations = copy.deepcopy(locations)
        self.capacities = copy.deepcopy(capacities)
        self.distance_matrix = []
        for from_index in range(len(self.locations)):
            distance_vector = []
            for to_index in range(len(self.locations)):
                distance_vector.append(calculate_distance(locations[from_index], locations[to_index]))
            self.distance_matrix.append(distance_vector)
        self.total_customer_capacities = 0
        for capacity in capacities[1:]:
            self.total_customer_capacities += capacity
        self.change_at = [0] * (len(self.locations) + 1)
        self.no_improvement_at = {}
        self.num_solutions = 0
        self.num_traversed = np.zeros((len(locations), len(locations)))
        self.distance_hashes = set()

    def record_solution(self, solution, distance):
        self.num_solutions += 1.0 / distance
        for path in solution:
            if len(path) > 2:
                for to_index in range(1, len(path)):
                    #TODO: change is needed for asymmetric cases.
                    self.num_traversed[path[to_index - 1]][path[to_index]] += 1.0 / distance
                    self.num_traversed[path[to_index]][path[to_index - 1]] += 1.0 / distance
                    # for index_in_the_same_path in range(to_index + 1, len(path)):
                    #     self.num_traversed[path[index_in_the_same_path]][path[to_index]] += 1
                    #     self.num_traversed[path[to_index]][path[index_in_the_same_path]] += 1

    def add_distance_hash(self, distance_hash):
        self.distance_hashes.add(distance_hash)

    def get_location(self, index):
        return self.locations[index]

    def get_capacity(self, index):
        return self.capacities[index]

    def get_capacity_ratio(self):
        return self.total_customer_capacities / float(self.get_capacity(0))

    def get_num_customers(self):
        return len(self.locations) - 1

    def get_distance(self, from_index, to_index):
        return self.distance_matrix[from_index][to_index]

    def get_frequency(self, from_index, to_index):
        return self.num_traversed[from_index][to_index] / (1.0 + self.num_solutions)

    def reset_change_at_and_no_improvement_at(self):
        self.change_at = [0] * (len(self.locations) + 1)
        self.no_improvement_at = {}

    def mark_change_at(self, step, path_indices):
        for path_index in path_indices:
            self.change_at[path_index] = step

    def mark_no_improvement(self, step, action, index_first, index_second=-1, index_third=-1):
        key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
        self.no_improvement_at[key] = step

    def should_try(self, action, index_first, index_second=-1, index_third=-1):
        key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
        no_improvement_at = self.no_improvement_at.get(key, -1)
        return self.change_at[index_first] >= no_improvement_at or \
               self.change_at[index_second] >= no_improvement_at or \
               self.change_at[index_third] >= no_improvement_at


def get_random_capacities(n):
    capacities = [0] * n
    depot_capacity_map = {
        10: 20,
        20: 30,
        50: 40,
        100: 50
    }
    capacities[0] = depot_capacity_map.get(n - 1, 50)
    for i in range(1, n):
        capacities[i] = np.random.randint(9) + 1
    return capacities

def generate_problem():
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    seed += 1

    test_points = 20
    num_sample_points = test_points + 1
    locations = np.random.uniform(size=(num_sample_points, 2))
    capacities = get_random_capacities(len(locations))
    problem = Problem(locations, capacities)
    return problem

x = generate_problem()
print(x)