import random
import copy
import math
from Utils import objective_function, check_best_position_change, check_first_position_change, check_best_tour_spot, \
    beam_check_best_position_change


def remove_insert(pdp, solution, op):
    solution = copy.copy(solution)
    op, op2 = op
    solution, removed_calls = op(pdp, solution)
    if removed_calls == [None]:
        new_solution, new_cost = insert_single_best(pdp, solution)
    else:
        new_solution, new_cost = op2(pdp, solution, removed_calls)
    return new_solution, new_cost


# Remove operators
# ==================================================================================================================

def remove_calls(solution, r_calls):
    temp_solution = [e for e in solution if math.ceil(e/2) not in r_calls]
    return temp_solution, r_calls


def remove_single_best(pdp, solution):
    return solution, [None]


def remove_close_calls(pdp, solution):
    n_remove = random.randint(min(pdp.n_calls, 2), min(pdp.n_calls, 10))
    dist_set = pdp.distances[n_remove]
    len_dist_set = len(dist_set)
    ind = random.randint(0, len_dist_set-1)
    ds = dist_set[ind]
    remove_calls(solution, set(ds))


def remove_longest_tour_deviation(pdp, solution):
    tour_deviation = []
    sol = [0] + solution + [solution[-1]]
    for i, e in enumerate(sol[1:-1], 1):
        a = pdp.dist_matrix[sol[i-1], e]
        b = pdp.dist_matrix[e, sol[i+1]]
        c = pdp.dist_matrix[sol[i-1], sol[i+1]]
        tour_deviation.append((e, a+b-c))
    tour_deviation = sorted(tour_deviation, key=lambda x: x[1])
    n_remove = random.randint(min(pdp.n_calls, 2), min(pdp.n_calls, 5))
    r_calls = set([math.ceil(x[0]/2) for x in tour_deviation[:n_remove]])
    return remove_calls(solution, r_calls)


def remove_tour_neighbors(pdp, solution):
    len_solution = len(solution)
    len_remove_segment = random.randint(min(len_solution, 2), min(len_solution, 5))
    start_segment = random.randint(0, len_solution-len_remove_segment)
    remove_segment = solution[start_segment:start_segment+len_remove_segment]
    r_calls = set([math.ceil(r/2) for r in remove_segment])
    return remove_calls(solution, r_calls)


def remove_xs(pdp, solution):
    r_calls = set(random.sample(range(1, pdp.n_calls + 1), random.randint(min(pdp.n_calls, 2), min(pdp.n_calls, 5))))
    return remove_calls(solution, r_calls)


def remove_s(pdp, solution):
    r_calls = set(random.sample(range(1, pdp.n_calls + 1), random.randint(min(pdp.n_calls, 5), min(pdp.n_calls, 10))))
    return remove_calls(solution, r_calls)


def remove_m(pdp, solution):
    r_calls = set(random.sample(range(1, pdp.n_calls + 1), random.randint(min(pdp.n_calls, 10), min(pdp.n_calls, 20))))
    return remove_calls(solution, r_calls)


def remove_l(pdp, solution):
    r_calls = set(random.sample(range(1, pdp.n_calls + 1), random.randint(min(pdp.n_calls, 20), min(pdp.n_calls, 30))))
    return remove_calls(solution, r_calls)


def remove_xl(pdp, solution):
    r_calls = set(random.sample(range(1, pdp.n_calls + 1), random.randint(min(pdp.n_calls, 30), min(pdp.n_calls, 40))))
    return remove_calls(solution, r_calls)


# Insert operators
# ==================================================================================================================


def insert_first(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_first_position_change(pdp, solution, i_call)
        solution.insert(ij[0], i_call*2-1)
        solution.insert(ij[1], i_call*2)
    cost = objective_function(pdp, solution)
    #assert cost != float('inf')
    #assert len(solution) == 50
    return solution, cost


def insert_greedy(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, solution, i_call)
        solution.insert(ij[0], i_call*2-1)
        solution.insert(ij[1], i_call*2)
    cost = objective_function(pdp, solution)
    #assert cost != float('inf')
    #assert len(solution) == 50
    return solution, cost


def insert_beam_search(pdp, solution, removed_calls, beam_width=10, search_width=5):
    cost = objective_function(pdp, solution)
    beam = [(solution, cost)]

    for i_call in removed_calls:
        first = i_call*2-1
        second = i_call*2
        next_beam = []
        for sol, cost in beam:
            i_j_cost_list = beam_check_best_position_change(pdp, sol, i_call, search_width)
            for i, j, add_cost in i_j_cost_list:
                new_sol = copy.copy(sol)
                new_sol.insert(i, first)
                new_sol.insert(j, second)
                next_beam.append((new_sol, cost+add_cost))
        next_beam = sorted(next_beam, key=lambda x: x[1])[:beam_width]  # beam width
        beam = next_beam

    solution, cost = beam[0]
    cost2 = objective_function(pdp, solution)
    #assert round(cost, 5) == round(cost2, 5)
    #assert cost != float('inf')
    #assert len(solution) == 50
    return solution, cost2


def insert_tour(pdp, solution, removed_calls):
    temp_sol = []
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, temp_sol, i_call)
        temp_sol.insert(ij[0], i_call * 2 - 1)
        temp_sol.insert(ij[1], i_call * 2)

    ind = check_best_tour_spot(pdp, solution, temp_sol)
    new_solution = solution[:ind] + temp_sol + solution[ind:]
    cost = objective_function(pdp, new_solution)
    #assert cost != float('inf')
    #assert len(new_solution) == 50
    return new_solution, cost


def insert_single_best(pdp, solution):
    min_cost = float('inf')
    best_change = (None, None, None)
    for i_call in range(1, pdp.n_calls+1):
        temp_solution = [e for e in solution if math.ceil(e/2) != i_call]
        cost = objective_function(pdp, temp_solution)
        ij, add_cost = check_best_position_change(pdp, temp_solution, i_call)
        if cost+add_cost < min_cost:
            min_cost = cost+add_cost
            best_change = (i_call, ij[0], ij[1])
    solution = [e for e in solution if math.ceil(e/2) != best_change[0]]
    solution.insert(best_change[1], best_change[0]*2-1)
    solution.insert(best_change[2], best_change[0]*2)
    # assert cost != float('inf')
    #assert len(solution) == 50
    return solution, min_cost
