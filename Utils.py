import numpy as np
from random import shuffle, sample
EPSILON = 1e-5


# euclidean distance
def get_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def get_weights(pdp, solution):
    current_weight = 0
    weights = [0]
    for e in solution:
        current_weight += pdp.capacities[e-1]
        weights.append(current_weight)
    weights.append(0)
    return weights


def embed_solution(pdp, solution):
    embedded_solution = np.zeros((len(solution), 11))
    weights = get_weights(pdp, solution)
    solution = [0] + solution[:] + [solution[-1]]
    for index in range(1, pdp.size):
        customer = solution[index]
        embedded_input = []
        embedded_input.append(pdp.capacities[customer-1])  # INSTANCE : capacity
        embedded_input.extend(pdp.locations[customer])  # INSTANCE : location
        embedded_input.append(1 - weights[index])  # SOLUTION : current free capacity
        embedded_input.extend(pdp.locations[solution[index - 1]])  # SOLUTION : location of prev
        embedded_input.extend(pdp.locations[solution[index + 1]])  # SOLUTION : location of next
        embedded_input.append(pdp.dist_matrix[solution[index - 1], customer])  # SOLUTION : dist from prev to node
        embedded_input.append(pdp.dist_matrix[customer, solution[index + 1]])  # SOLUTION : dist from node to next
        embedded_input.append(pdp.dist_matrix[solution[index - 1], solution[index + 1]])  # SOLUTION : dist if current node removed
        for embedded_input_index in range(len(embedded_input)):
            embedded_solution[customer - 1, embedded_input_index] = embedded_input[embedded_input_index]
    return embedded_solution


# Calculates cost and feasibility of the solution:
def objective_function(pdp, solution):
    cost = 0
    current_weight = 0
    current_node = 0
    visited = [False] * (pdp.size+1)
    for e in solution:
        if e % 2 == 0 and not visited[e-1]:
            return float('inf')  # infeasible due to delivery before pickup
        visited[e] = True
        cost += pdp.dist_matrix[current_node, e]
        current_weight += pdp.capacities[e-1]
        current_node = e
        if current_weight > 1 + EPSILON:
            return float('inf')  # infeasible due to capacity constraint
    return cost


def check_best_position_change(pdp, solution, i_call):
    len_solution = len(solution)
    i_call_weight = pdp.capacities[int(i_call*2)-2]

    if len_solution > 0:
        cumulative_weight = [0] + [None]*len(solution)
        for i, e in enumerate(solution, 1):
            cumulative_weight[i] = cumulative_weight[i-1] + pdp.capacities[e-1]
    best_ij = None
    best_cost = float('inf')
    for i in range(len_solution+1):

        # SIZE CHECK:
        if len_solution > 0 and cumulative_weight[i] + i_call_weight > 1 + EPSILON:
            continue  # if inserting i_call here already makes the vehicle overloaded

        for j in range(i+1, len_solution+2):

            # SIZE CHECK:
            if len_solution > 0 and cumulative_weight[j - 1] + i_call_weight > 1 + EPSILON:
                break  # Go to next i

            if i + 1 == j:
                if i == 0 and j == len_solution + 1:
                    R1 = 0
                    A1 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = 0
                elif i == 0:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A1 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = pdp.dist_matrix[pdp.calls[i_call][1], solution[i]]
                elif j == len_solution + 1:
                    R1 = 0
                    A1 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = 0
                else:
                    R1 = pdp.dist_matrix[solution[i-1], solution[i]]
                    A1 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = pdp.dist_matrix[pdp.calls[i_call][1], solution[i]]
                add_cost = -R1 + A1 + A2 + A3
            else:
                if i == 0 and j == len_solution + 1:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A11 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = 0
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = 0
                elif i == 0:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A11 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = pdp.dist_matrix[solution[j-2], solution[j-1]]
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = pdp.dist_matrix[pdp.calls[i_call][1], solution[j-1]]
                elif j == len_solution + 1:
                    R1 = pdp.dist_matrix[solution[i-1], solution[i]]
                    A11 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = 0
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = 0
                else:
                    R1 = pdp.dist_matrix[solution[i-1], solution[i]]
                    A11 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = pdp.dist_matrix[solution[j-2], solution[j-1]]
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = pdp.dist_matrix[pdp.calls[i_call][1], solution[j-1]]
                add_cost = - R1 - R2 + A11 + A12 + A21 + A22

            # CHECK IF NEW BEST POSITION:
            if add_cost < best_cost:
                best_ij = (i, j)
                best_cost = add_cost

    return best_ij, best_cost


def check_first_position_change(pdp, solution, i_call):
    len_solution = len(solution)
    if len_solution == 0:
        return (0, 1), None
    i_call_weight = pdp.capacities[int(i_call*2)-2]

    if len_solution > 0:
        cumulative_weight = [0] + [None]*len(solution)
        for i, e in enumerate(solution, 1):
            cumulative_weight[i] = cumulative_weight[i-1] + pdp.capacities[e-1]

    first = list(range(len_solution+1))
    shuffle(first)
    potentials = None
    for i in first:
        if potentials:
            return sample(potentials, 1)[0], None
        if len_solution > 0 and cumulative_weight[i] + i_call_weight > 1 + EPSILON:  # SIZE CHECK:
            continue
        potentials = []
        for j in range(i+1, len_solution+2):
            if len_solution > 0 and cumulative_weight[j - 1] + i_call_weight > 1 + EPSILON:  # SIZE CHECK:
                break
            potentials.append((i, j))


def check_best_tour_spot(pdp, solution, temp_sol):
    len_solution = len(solution)
    a, b = temp_sol[0], temp_sol[-1]
    if len_solution == 0:
        return 0

    temp_sol_cumulative_weight = [0] + [None] * len(temp_sol)
    for i, e in enumerate(temp_sol, 1):
        temp_sol_cumulative_weight[i] = temp_sol_cumulative_weight[i - 1] + pdp.capacities[e - 1]
    max_weight = max(temp_sol_cumulative_weight)

    if len_solution > 0:
        cumulative_weight = [0] + [None]*len_solution
        for i, e in enumerate(solution, 1):
            cumulative_weight[i] = cumulative_weight[i-1] + pdp.capacities[e-1]

    best_ind = None
    best_cost = float('inf')

    for i in range(len_solution+1):

        # SIZE CHECK:
        if len_solution > 0 and cumulative_weight[i] + max_weight > 1 + EPSILON:
            continue  # if inserting i_call here already makes the vehicle overloaded

        if i == 0:
            R1 = pdp.dist_matrix[0, solution[i]]
            A1 = pdp.dist_matrix[0, a]
            A2 = pdp.dist_matrix[b, solution[i]]
        elif i == len_solution:
            R1 = 0
            A1 = pdp.dist_matrix[solution[i-1], a]
            A2 = 0
        else:
            R1 = pdp.dist_matrix[solution[i-1], solution[i]]
            A1 = pdp.dist_matrix[solution[i-1], a]
            A2 = pdp.dist_matrix[b, solution[i]]
        add_cost = -R1 + A1 + A2

        # CHECK IF NEW BEST POSITION:
        if add_cost < best_cost:
            best_ind = i
            best_cost = add_cost

    return best_ind


def beam_check_best_position_change(pdp, solution, i_call, search_width):
    len_solution = len(solution)
    i_call_weight = pdp.capacities[int(i_call * 2) - 2]

    if len_solution > 0:
        cumulative_weight = [0] + [None] * len(solution)
        for i, e in enumerate(solution, 1):
            cumulative_weight[i] = cumulative_weight[i - 1] + pdp.capacities[e - 1]

    i_j_cost_list = []

    for i in range(len_solution + 1):

        # SIZE CHECK:
        if len_solution > 0 and cumulative_weight[i] + i_call_weight > 1 + EPSILON:
            continue  # if inserting i_call here already makes the vehicle overloaded

        for j in range(i + 1, len_solution + 2):

            # SIZE CHECK:
            if len_solution > 0 and cumulative_weight[j - 1] + i_call_weight > 1 + EPSILON:
                break  # Go to next i

            if i + 1 == j:
                if i == 0 and j == len_solution + 1:
                    R1 = 0
                    A1 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = 0
                elif i == 0:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A1 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = pdp.dist_matrix[pdp.calls[i_call][1], solution[i]]
                elif j == len_solution + 1:
                    R1 = 0
                    A1 = pdp.dist_matrix[solution[i - 1], pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = 0
                else:
                    R1 = pdp.dist_matrix[solution[i - 1], solution[i]]
                    A1 = pdp.dist_matrix[solution[i - 1], pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = pdp.dist_matrix[pdp.calls[i_call][1], solution[i]]
                add_cost = -R1 + A1 + A2 + A3
            else:
                if i == 0 and j == len_solution + 1:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A11 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = 0
                    A21 = pdp.dist_matrix[solution[j - 2], pdp.calls[i_call][1]]
                    A22 = 0
                elif i == 0:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A11 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = pdp.dist_matrix[solution[j - 2], solution[j - 1]]
                    A21 = pdp.dist_matrix[solution[j - 2], pdp.calls[i_call][1]]
                    A22 = pdp.dist_matrix[pdp.calls[i_call][1], solution[j - 1]]
                elif j == len_solution + 1:
                    R1 = pdp.dist_matrix[solution[i - 1], solution[i]]
                    A11 = pdp.dist_matrix[solution[i - 1], pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = 0
                    A21 = pdp.dist_matrix[solution[j - 2], pdp.calls[i_call][1]]
                    A22 = 0
                else:
                    R1 = pdp.dist_matrix[solution[i - 1], solution[i]]
                    A11 = pdp.dist_matrix[solution[i - 1], pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = pdp.dist_matrix[solution[j - 2], solution[j - 1]]
                    A21 = pdp.dist_matrix[solution[j - 2], pdp.calls[i_call][1]]
                    A22 = pdp.dist_matrix[pdp.calls[i_call][1], solution[j - 1]]
                add_cost = - R1 - R2 + A11 + A12 + A21 + A22

            i_j_cost_list.append((i, j, add_cost))

    return sorted(i_j_cost_list, key=lambda x: x[2])[:search_width]

