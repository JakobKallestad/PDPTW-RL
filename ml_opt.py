import argparse
import collections
import copy
import datetime
import math
import numpy as np
import os
import pickle
import random
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorboardX import SummaryWriter

from sequence_encoder import encode_seq, embed_seq

from Utils import get_distance, embed_solution, objective_function
from Operators import remove_insert, remove_single_best, remove_longest_tour_deviation, remove_tour_neighbors,\
    remove_xs, remove_s, remove_m, remove_l, remove_xl, insert_first, insert_greedy, insert_beam_search, insert_tour


EPSILON = 1e-6
writer = SummaryWriter('logs_7')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_config(args=None):
    parser = argparse.ArgumentParser(description="Meta optimization")
    parser.add_argument('--epoch_size', type=int, default=5120000, help='Epoch size')

    parser.add_argument('--num_feedforward_units', type=int, default=128, help="number of feedforward units")
    parser.add_argument('--problem', default='vrp', help="the problem to be solved, {tsp, vrp}")
    parser.add_argument('--train_operators', type=str2bool, nargs='?', const=True, default=False, help="")

    parser.add_argument('--num_training_points', type=int, default=100, help="size of the problem for training")
    parser.add_argument('--num_test_points', type=int, default=100, help="size of the problem for testing")
    parser.add_argument('--num_episode', type=int, default=40000, help="number of training episode")
    parser.add_argument('--max_num_rows', type=int, default=2000000, help="")
    # parser.add_argument('--num_paths_to_ruin', type=int, default=2, help="")
    parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
    parser.add_argument('--max_rollout_steps', type=int, default=20000, help="maximum rollout steps")
    parser.add_argument('--max_rollout_seconds', type=int, default=1000, help="maximum rollout time in seconds")
    parser.add_argument('--use_rl_loss', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--use_attention_embedding', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--epsilon_greedy', type=float, default=0.05, help="")
    parser.add_argument('--sample_actions_in_rollout', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--num_active_learning_iterations', type=int, default=1, help="")
    parser.add_argument('--max_no_improvement', type=int, default=6, help="")
    parser.add_argument('--debug_mode', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--debug_steps', type=int, default=1, help="")
    parser.add_argument('--num_actions', type=int, default=32, help="dimension of action space")
    # parser.add_argument('--max_num_customers_to_shuffle', type=int, default=20, help="")
    parser.add_argument('--problem_seed', type=int, default=1, help="problem generating seed")
    # parser.add_argument('--input_embedded_trip_dim', type=int, default=9, help="")
    parser.add_argument('--input_embedded_trip_dim_2', type=int, default=11, help="")
    parser.add_argument('--num_embedded_dim_1', type=int, default=64, help="")
    parser.add_argument('--num_embedded_dim_2', type=int, default=64, help="dim")
    parser.add_argument('--discount_factor', type=float, default=1.0, help="discount factor of policy network")
    parser.add_argument('--policy_learning_rate', type=float, default=0.001, help="learning rate of policy network")
    parser.add_argument('--hidden_layer_dim', type=int, default=64, help="dimension of hidden layer in policy network")
    parser.add_argument('--num_history_action_use', type=int, default=0, help="number of history actions used in the representation of current state")
    parser.add_argument('--step_interval', type=int, default=500)

    # './rollout_model_1850.ckpt'
    parser.add_argument('--model_to_restore', type=str, default=None, help="")
    parser.add_argument('--max_num_training_epsisodes', type=int, default=10000000, help="")

    parser.add_argument('--max_points_per_trip', type=int, default=15, help="upper bound of number of point in one trip")
    parser.add_argument('--max_trips_per_solution', type=int, default=15, help="upper bound of number of trip in one solution")

    config = parser.parse_args(args)
    return config


config = get_config()
if config.max_no_improvement is None:
    config.max_no_improvement = config.num_actions



CAPACITY_MAP = {
    10: 10,  # 20,
    20: 15,  # 30,
    50: 20,  # 40,
    100: 25,  # 50
}


# OPERATORS
remove_operators = [remove_single_best, remove_longest_tour_deviation, remove_tour_neighbors,
                        remove_xs, remove_s, remove_m, remove_l, remove_xl]
insert_operators = [insert_first, insert_greedy, insert_beam_search, insert_tour]


class PDP:
    def __init__(self, size, n_calls, locations, capacities, calls, dist_matrix):
        self.size = size
        self.n_calls = n_calls
        self.locations = locations
        self.capacities = capacities
        self.calls = calls
        self.dist_matrix = dist_matrix
        self.initialize_close_calls()

        # self.change_at = [0] * (len(self.locations) + 1)
        # self.no_improvement_at = {}
        # self.num_solutions = 0
        # self.num_traversed = np.zeros((len(locations), len(locations)))
        # self.distance_hashes = set()

    def save_problem(self):
        pass

    def initialize_close_calls(self):
        self.distances = self.calculate_close_calls()

    def calculate_close_calls(self):
        max_sim_size = 21
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

    #
    # def reset_change_at_and_no_improvement_at(self):
    #     self.change_at = [0] * (len(self.locations) + 1)
    #     self.no_improvement_at = {}
    #
    # def mark_no_improvement(self, step, action, index_first, index_second=-1, index_third=-1):
    #     key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
    #     self.no_improvement_at[key] = step
    #
    # def should_try(self, action, index_first, index_second=-1, index_third=-1):
    #     key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
    #     no_improvement_at = self.no_improvement_at.get(key, -1)
    #     return self.change_at[index_first] >= no_improvement_at or \
    #            self.change_at[index_second] >= no_improvement_at or \
    #            self.change_at[index_third] >= no_improvement_at


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
    pdp.save_problem()
    return pdp


def load_pdp_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def improve_solution_by_action(pdp, solution, action):
    solution = copy.copy(solution)

    # ENV ACT
    remove_op = remove_operators[action // 4]
    insert_op = insert_operators[action % 4]
    op = (remove_op, insert_op)
    solution, cost = remove_insert(pdp, solution, op)
    return solution


# def dense_to_one_hot(labels_dense, num_training_points):
#   """Convert class labels from scalars to one-hot vectors."""
#   num_labels = labels_dense.shape[0]
#   index_offset = np.arange(num_labels) * config.num_training_points
#   labels_one_hot = np.zeros((num_labels, config.num_training_points))
#   labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
#   return labels_one_hot
#
#
# def reshape_input(input, x, y, z):
#     return np.reshape(input, (x, y, z))


is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)


# def sample_next_index(to_indices, adjusted_distances):
#     if len(to_indices) == 0:
#         return 0
#     adjusted_probabilities = np.asarray([1.0 / max(d, EPSILON) for d in adjusted_distances])
#     adjusted_probabilities /= np.sum(adjusted_probabilities)
#     return np.random.choice(to_indices, p=adjusted_probabilities)
#     # return to_indices[np.argmax(adjusted_probabilities)]


# Very simple now. Look into how to perturb the solution in a more meaningful way
def construct_solution(pdp, existing_solution=None, step=0):
    #pdp.reset_change_at_and_no_improvement_at()
    return [i for i in range(1, pdp.size + 1)]  # Try to perturb solution in a meaningful way!


def get_num_points(config):
    if config.model_to_restore is None:
        return config.num_training_points
    else:
        return config.num_test_points


ATTENTION_ROLLOUT, LSTM_ROLLOUT = True, False #False, True


# def embedding_net_nothing(input_):
#     return input_


def embedding_net_2(input_):
    with tf.variable_scope("embedding_net"):
        architecture_type = 0
        if architecture_type == 0:
            x = embed_seq(input_seq=input_, from_=config.input_embedded_trip_dim_2, to_=config.num_embedded_dim_1, is_training=True,
                          BN=True, initializer=tf.contrib.layers.xavier_initializer())

            layer_attention = encode_seq(input_seq=x, input_dim=config.num_embedded_dim_1, num_stacks=1, num_heads=8,
                                         num_neurons=64, is_training=True, dropout_rate=0.1)
            # layer_attention = tf.reshape(layer_attention, [-1, (config.num_training_points) * config.num_embedded_dim_1])
            # layer_2 = tf.contrib.layers.fully_connected(layer_attention, config.num_embedded_dim_2, activation_fn=tf.nn.relu)
            # layer_2 = tf.nn.dropout(layer_2, keep_prob)
            layer_2 = tf.reduce_sum(layer_attention, axis=1)
        else:
            #TODO:
            x = embed_seq(input_seq=input_, from_=config.input_embedded_trip_dim_2, to_=config.num_embedded_dim_1, is_training=True, BN=False, initializer=tf.contrib.layers.xavier_initializer())
            x = tf.reduce_sum(x, axis=1)
            layer_2 = tf.nn.relu(x)
    return layer_2


# def embed_trip(trip, points_in_trip):
#     trip_prev = np.vstack((trip[-1], trip[:-1]))
#     trip_next = np.vstack((trip[1:], trip[0]))
#     distance_from_prev = np.reshape(np.linalg.norm(trip_prev - trip, axis=1), (points_in_trip, 1))
#     distance_to_next = np.reshape(np.linalg.norm(trip - trip_next, axis=1), (points_in_trip, 1))
#     distance_from_to_next = np.reshape(np.linalg.norm(trip_prev - trip_next, axis=1), (points_in_trip, 1))
#     trip_with_additional_information = np.hstack((trip_prev, trip, trip_next, distance_from_prev, distance_to_next, distance_from_to_next))
#     return trip_with_additional_information


def embedding_net_attention(input_):
    with tf.variable_scope("attention_embedding"):
        x = embed_seq(input_seq=input_, from_=config.num_embedded_dim, to_=128, is_training=True, BN=True, initializer=tf.contrib.layers.xavier_initializer())
        layer_attention = encode_seq(input_seq=x, input_dim=128, num_stacks=3, num_heads=16, num_neurons=512, is_training=True, dropout_rate=0.1)
        layer_attention = tf.reshape(layer_attention, [-1, config.max_trips_per_solution * 128])
        layer_2 = tf.contrib.layers.fully_connected(layer_attention, config.num_embedded_dim, activation_fn=tf.nn.relu)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
    return layer_2


# def embed_solution(problem, solution):
#     embedded_solution = np.zeros((config.max_trips_per_solution, config.max_points_per_trip, config.input_embedded_trip_dim))
#     n_trip = len(solution)
#     for trip_index in range(min(config.max_trips_per_solution, n_trip)):
#         trip = solution[trip_index]
#         truncated_trip_length = np.minimum(config.max_points_per_trip, len(trip) - 1)
#         if truncated_trip_length > 1:
#             points_with_coordinate = np.zeros((truncated_trip_length, 2))
#             for point_index in range(truncated_trip_length):
#                 points_with_coordinate[point_index] = problem.get_location(trip[point_index])
#             embedded_solution[trip_index, :truncated_trip_length] = copy.deepcopy(embed_trip(points_with_coordinate, truncated_trip_length))
#     return embedded_solution


# def embed_solution_with_nothing(problem, solution):
#     embedded_solution = np.zeros((config.max_trips_per_solution, config.max_points_per_trip, config.input_embedded_trip_dim))
#     return embedded_solution


# def embed_solution_with_attention(problem, solution):
#     embedded_solution = np.zeros((config.num_training_points, config.input_embedded_trip_dim_2))
#
#     for path in solution:
#         if len(path) == 2:
#             continue
#         n = len(path) - 1
#         consumption = calculate_consumption(problem, path)
#         for index in range(1, n):
#             customer = path[index]
#             embedded_input = []
#             embedded_input.append(problem.get_capacity(customer))
#             embedded_input.extend(problem.get_location(customer))
#             embedded_input.append(problem.get_capacity(0) - consumption[-1])
#             embedded_input.extend(problem.get_location(path[index - 1]))
#             embedded_input.extend(problem.get_location(path[index + 1]))
#             embedded_input.append(problem.get_distance(path[index - 1], customer))
#             embedded_input.append(problem.get_distance(customer, path[index + 1]))
#             embedded_input.append(problem.get_distance(path[index - 1], path[index + 1]))
#             for embedded_input_index in range(len(embedded_input)):
#                 embedded_solution[customer - 1, embedded_input_index] = embedded_input[embedded_input_index]
#     return embedded_solution


TEST_X = tf.placeholder(tf.float32, [None, config.num_training_points, config.input_embedded_trip_dim_2])
embedded_x = embedding_net_2(TEST_X)
env_observation_space_n = config.num_history_action_use * 2 + 5
action_labels_placeholder = tf.placeholder("float", [None, config.num_actions - 1])


class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=config.policy_learning_rate, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.states = tf.placeholder(tf.float32, [None, env_observation_space_n], "states")
            if config.use_attention_embedding:
                full_states = tf.concat([self.states, embedded_x], axis=1)
            else:
                full_states = self.states

            self.hidden1 = tf.contrib.layers.fully_connected(
                inputs=full_states,
                num_outputs=config.hidden_layer_dim,
                activation_fn=tf.nn.relu)
            self.logits = tf.contrib.layers.fully_connected(
                inputs=self.hidden1,
                num_outputs=config.num_actions - 1,
                activation_fn=None)
            #https://stackoverflow.com/questions/33712178/tensorflow-nan-bug?newreg=c7e31a867765444280ba3ca50b657a07
            self.action_probs = tf.clip_by_value(tf.nn.softmax(self.logits), 1e-10, 1.0)

            # training part of graph
            self.actions = tf.placeholder(tf.int32, [None], name="actions")
            self.advantages = tf.placeholder(tf.float32, [None], name="advantages")
            log_prob = tf.log(self.action_probs)
            indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self.actions
            act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)
            self.loss = -tf.reduce_sum(act_prob * self.advantages)
            # self.loss = -tf.reduce_mean(act_prob * self.advantages)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

            # Define loss and optimizer
            # Define loss and optimizer
            self.sl_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=action_labels_placeholder))
            self.sl_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.sl_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.sl_train_op = self.sl_optimizer.minimize(self.sl_loss)
            # Training accuracy
            correct_pred = tf.equal(tf.argmax(self.action_probs, 1), tf.argmax(action_labels_placeholder, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def predict(self, states, test_x, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: states, TEST_X: test_x, keep_prob: 1.0}
        return sess.run(self.action_probs, feed_dict)

    def update(self, states, test_x, advantages, actions, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: states, self.advantages: advantages, self.actions: actions, TEST_X:test_x, keep_prob:1.0}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def train(self, states, test_x, action_labels, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: states, TEST_X:test_x, action_labels_placeholder: action_labels, keep_prob:1.0}
        _, loss, accuracy = sess.run([self.sl_train_op, self.sl_loss, self.accuracy], feed_dict)
        return loss, accuracy


previous_solution = None
initial_solution = None
best_solution = None


def env_act(pdp, solution, action):
    solution = improve_solution_by_action(pdp, solution, action)
    cost = objective_function(pdp, solution)
    if cost == float('inf'):
        print('Invalid solution!')
    return solution, cost


action_timers = [0.0] * (config.num_actions * 2)


def env_generate_state(min_distance=None, state=None, action=None, distance=None, next_distance=None):
    if state is None: #or action == 0:
        next_state = [0.0, 0.0, 0]
        for _ in range(config.num_history_action_use):
            next_state.append(0.0)
            next_state.append(0)
        next_state.append(0.0)
        next_state.append(0)
    else:
        delta = next_distance - distance
        if delta < -EPSILON:
            delta_sign = -1.0
        else:
            delta_sign = 1.0
        next_state = [0.0, next_distance - min_distance, delta]
        if config.num_history_action_use != 0:
            next_state.extend(state[(-config.num_history_action_use * 2):])
        next_state.append(delta_sign)
        next_state.append(action)
    return next_state


def env_step(step, state, pdp, min_distance, solution, distance, action, record_time=True):
    start_timer = datetime.datetime.now()
    solution, next_distance = env_act(pdp, solution, action)
    next_state = env_generate_state(min_distance, state, action, distance, next_distance)
    reward = distance - next_distance
    end_timer = datetime.datetime.now()
    if record_time:
        action_timers[action * 2] += 1
        action_timers[action * 2 + 1] += (end_timer - start_timer).total_seconds()
    done = (datetime.datetime.now() - env_start_time).total_seconds() >= config.max_rollout_seconds
    return next_state, reward, done, solution, next_distance


def format_print(value):
    return round(float(value), 2)


def format_print_array(values):
    results = []
    for value in values:
        results.append(format_print(value))
    return results


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print ([str(i.name) for i in not_initialized_vars])
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


# def should_restart(min_distance, distance, no_improvement):
#     if no_improvement >= config.max_no_improvement:
#         return True
#     if no_improvement <= config.max_no_improvement - 1:
#         return False
#     percentage_over = round((distance / min_distance - 1.0) * 100)
#     upper_limits = [20, 10, 5, 2]
#     return percentage_over >= upper_limits[no_improvement - 2]


gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
with tf.Session(config=gpu_config) as sess:
    policy_estimator = PolicyEstimator()
    initialize_uninitialized(sess)
    print(sess.run(tf.report_uninitialized_variables()))
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print("Variable={}, Shape={}".format(k, v.shape))
    sys.stdout.flush()
    saver = tf.train.Saver()
    if config.model_to_restore is not None:
        saver.restore(sess, config.model_to_restore)

    distances = []
    steps = []
    consolidated_distances, consolidated_steps = [], []
    timers = []
    num_checkpoint = int(config.max_rollout_steps/config.step_interval)
    step_record = np.zeros((2, num_checkpoint))
    distance_record = np.zeros((2, num_checkpoint))
    start = datetime.datetime.now()
    seed = config.problem_seed
    tf.set_random_seed(seed)

    Transition = collections.namedtuple("Transition", ["state", "trip", "next_distance", "action", "reward", "next_state", "done"])
    for index_sample in range(config.num_episode):
        states = []
        trips = []
        actions = []
        advantages = []
        action_labels = []
        if index_sample > 0 and index_sample % config.debug_steps == 0:
            if True:  #not config.use_random_rollout:
                formatted_timers = format_print_array(np.mean(np.asarray(timers), axis=0))
                count_timers = formatted_timers[4:][::2]
                time_timers = formatted_timers[4:][1::2]
                print('time ={}'.format('\t\t'.join([str(x) for x in time_timers])))
                print('count={}'.format('\t\t'.join([str(x) for x in count_timers])))
                start_active = ((len(distances) // config.num_active_learning_iterations) * config.num_active_learning_iterations)
                if start_active == len(distances):
                    start_active -= config.num_active_learning_iterations
                tail_distances = distances[start_active:]
                tail_steps = steps[start_active:]
                min_index = np.argmin(tail_distances)
                if config.num_active_learning_iterations == 1 or len(distances) % config.num_active_learning_iterations == 1:
                    consolidated_distances.append(tail_distances[min_index])
                    consolidated_steps.append(tail_steps[min_index] + min_index * config.max_rollout_steps)
                else:
                    consolidated_distances[-1] = tail_distances[min_index]
                    consolidated_steps[-1] = tail_steps[min_index] + min_index * config.max_rollout_steps
                print('index_sample={}, mean_distance={}, mean_step={}, tail_distance={}, last_distance={}, last_step={}, timers={}'.format(
                    index_sample,
                    format_print(np.mean(consolidated_distances)), format_print(np.mean(consolidated_steps)),
                    format_print(np.mean(consolidated_distances[max(0, len(consolidated_distances) - 1000):])),
                    format_print(consolidated_distances[-1]), consolidated_steps[-1],
                    formatted_timers[:4]
                ))
                sys.stdout.flush()
            else:
                formatted_timers = format_print_array(np.mean(np.asarray(timers), axis=0))
                for index in range(num_checkpoint):
                    print('rollout_num={}, index_sample={}, mean_distance={}, mean_step={}, last_distance={}, last_step={}, timers={}'.format(
                        (index + 1) * config.step_interval, index_sample, ((index_sample - 1) * distance_record[0, index] + distance_record[1, index]) / index_sample,
                        ((index_sample - 1) * step_record[0, index] + step_record[1, index]) / index_sample, distance_record[1, index],
                        step_record[1, index], formatted_timers[:4]
                    ))
                    step_record[0, index] = ((index_sample - 1) * step_record[0, index] + step_record[1, index]) / index_sample
                    distance_record[0, index] = ((index_sample - 1) * distance_record[0, index] + distance_record[1, index]) / index_sample
                sys.stdout.flush()

        pdp = generate_problem(size=config.num_training_points)
        solution = construct_solution(pdp)
        best_solution = copy.deepcopy(solution)

        embedded_trip = embed_solution(pdp, solution)
        min_distance = objective_function(pdp, solution)
        min_step = 0
        distance = min_distance

        state = env_generate_state()
        env_start_time = datetime.datetime.now()
        episode = []
        current_best_distances = []
        start_distance = distance
        current_distances = []
        start_distances = []

        inference_time = 0
        gpu_inference_time = 0
        env_act_time = 0
        no_improvement = 0
        loop_action = 0
        num_random_actions = 0
        num_improvements = 0
        num_best_improvements = 0

        for step in range(config.max_rollout_steps):
            start_timer = datetime.datetime.now()
            gpu_start_time = datetime.datetime.now()
            action_probs = policy_estimator.predict([state], [embedded_trip], sess)
            gpu_inference_time += (datetime.datetime.now() - gpu_start_time).total_seconds()
            action_probs = action_probs[0]
            history_action_probs = np.zeros(len(action_probs))
            action_prob_sum = 0.0
            for action_prob_index in range(len(action_probs)):
                action_prob_sum += action_probs[action_prob_index]
            for action_prob_index in range(len(action_probs)):
                action_probs[action_prob_index] /= action_prob_sum


            if config.use_rl_loss:
                states.append(state)
                trips.append(embedded_trip)

            #if (config.model_to_restore is not None and should_restart(min_distance, distance, no_improvement)) or no_improvement >= config.max_no_improvement:
            #    action = 0
            #    no_improvement = 0
            #else:
            if np.random.uniform() < config.epsilon_greedy:
                action = np.random.randint(config.num_actions - 1) + 1
                num_random_actions += 1
            else:
                if config.sample_actions_in_rollout:
                    action = np.random.choice(np.arange(len(action_probs)), p=action_probs) + 1
                else:
                    action = np.argmax(action_probs) + 1
            end_timer = datetime.datetime.now()
            inference_time += (end_timer - start_timer).total_seconds()
            start_timer = end_timer

            next_state, reward, done, next_solution, next_distance = env_step(step, state, pdp, min_distance, solution, distance, action)
            if next_distance >= distance - EPSILON:
                no_improvement += 1
            else:
                #TODO
                no_improvement = 0
                num_improvements += 1

            current_distances.append(distance)
            start_distances.append(start_distance)
            # if action == 0:
            #     start_distance = next_distance
            current_best_distances.append(min_distance)
            if next_distance < min_distance - EPSILON:
                min_distance = next_distance
                min_step = step
                best_solution = copy.deepcopy(next_solution)
                num_best_improvements += 1
            if (step + 1) % config.step_interval == 0:
                print('rollout_num={}, index_sample={}, min_distance={}, min_step={}'.format(
                    step + 1, index_sample, min_distance, min_step
                ))
                temp_timers = np.asarray(action_timers)
                temp_count_timers = temp_timers[::2]
                temp_time_timers = temp_timers[1::2]
                print('time ={}'.format('\t\t'.join([str(x) for x in temp_time_timers])))
                print('count={}'.format('\t\t'.join([str(x) for x in temp_count_timers])))
            if done:
                break

            episode.append(Transition(
                state=state, trip=copy.deepcopy(embedded_trip), next_distance=next_distance,
                action=action, reward=reward, next_state=next_state, done=done))
            state = next_state
            solution = next_solution
            embedded_trip = embed_solution(pdp, solution)
            distance = next_distance
            end_timer = datetime.datetime.now()
            env_act_time += (end_timer - start_timer).total_seconds()
            start_timer = end_timer

        start_timer = datetime.datetime.now()
        distances.append(min_distance)
        steps.append(min_step)
        if objective_function(pdp, best_solution) < float('inf'):
            print('solution={}'.format(best_solution))
        else:
            print('invalid solution')

        # TENSORBOARD:
        writer.add_scalar('min_distance', float(min_distance), index_sample)
        writer.add_scalar('min_step', float(min_step), index_sample)
        writer.add_scalar('num_improvement', num_improvements, index_sample)
        writer.add_scalar('num_best_improvement', num_best_improvements, index_sample)
        writer.add_scalar('num_steps', step, index_sample)


        future_best_distances = [0.0] * len(episode)
        future_best_distances[-1] = episode[len(episode) - 1].next_distance
        step = len(episode) - 2
        while step >= 0:
            if episode[step].action != 0:
                future_best_distances[step] = future_best_distances[step + 1] * config.discount_factor
            else:
                future_best_distances[step] = current_distances[step]
            step = step - 1

        historical_baseline = 0  #None
        for t, transition in enumerate(episode):
            # total_return = sum(config.discount_factor**i * future_transition.reward for i, future_transition in enumerate(episode[t:]))
            if historical_baseline is None:
                # if transition.action == 0:
                #     #TODO: dynamic updating of historical baseline, and state definition
                #     historical_baseline = -current_best_distances[t]
                #     # historical_baseline = 1/(current_best_distances[t] - 10)
                actions.append(0)
                advantages.append(0)
                continue
            # if transition.action == 0:
            #     historical_baseline = -current_distances[t]
            if transition.action >= 0:
                #total_return = transition.reward
                if transition.reward < EPSILON:
                    total_return = -1.0
                else:
                    total_return = 1.0
                #     total_return = min(transition.reward, 2.0)
                # total_return = start_distances[t] - future_best_distances[t]
                # total_return = min(total_return, 1.0)
                # total_return = max(total_return, -1.0)
                ######## total_return = -future_best_distances[t]  # this was the default one, jakob
                # total_return = 1/(future_best_distances[t] - 10)
            else:
                if transition.state[-1] != 0 and transition.state[-2] < 1e-6:
                    # if future_best_distances[t] < current_best_distances[t] - 1e-6:
                    total_return = 1.0
                else:
                    total_return = -1.0
                total_return = 0
                actions.append(0)
                advantages.append(0)
                continue
            # baseline_value = value_estimator.predict(states)
            # baseline_value = 0.0
            baseline_value = historical_baseline
            advantage = total_return - baseline_value
            actions.append(transition.action)
            advantages.append(advantage)
            # value_estimator.update(states, [[total_return]], sess)

        states = np.reshape(np.asarray(states), (-1, env_observation_space_n)).astype("float32")
        if config.use_attention_embedding:
            trips = np.reshape(np.asarray(trips), (-1, config.num_training_points, config.input_embedded_trip_dim_2)).astype("float32")
        actions = np.reshape(np.asarray(actions), (-1))
        advantages = np.reshape(np.asarray(advantages), (-1)).astype("float32")
        if config.use_rl_loss:
            print('num_random_actions={}'.format(num_random_actions))
            print('actions={}'.format(actions[:100]).replace('\n', ''))
            print('advantages={}'.format(advantages[:100]).replace('\n', ''))
            if config.model_to_restore is None and index_sample <= config.max_num_training_epsisodes:
                filtered_states = []
                filtered_trips = []
                filtered_advantages = []
                filtered_actions = []
                end = len(actions)+5  # 0
                for action_index in range(len(actions)):
                    # if actions[action_index] >= 0:
                    filtered_states.append(states[action_index])
                    filtered_trips.append(trips[action_index])
                    filtered_advantages.append(advantages[action_index])
                    filtered_actions.append(actions[action_index] - 1)
                    # else:
                    #     num_bad_steps = config.max_no_improvement
                    #     end = max(end, len(filtered_states) - num_bad_steps)
                    #     filtered_states = filtered_states[:end]
                    #     filtered_trips = filtered_trips[:end]
                    #     filtered_advantages = filtered_advantages[:end]
                    #     filtered_actions = filtered_actions[:end]
                filtered_states = filtered_states[:end]
                filtered_trips = filtered_trips[:end]
                filtered_advantages = filtered_advantages[:end]
                filtered_actions = filtered_actions[:end]
                num_states = len(filtered_states)
                if config.use_attention_embedding and num_states > config.batch_size:
                    downsampled_indices = np.random.choice(range(num_states), config.batch_size, replace=False)
                    filtered_states = np.asarray(filtered_states)[downsampled_indices]
                    filtered_trips = np.asarray(filtered_trips)[downsampled_indices]
                    filtered_advantages = np.asarray(filtered_advantages)[downsampled_indices]
                    filtered_actions = np.asarray(filtered_actions)[downsampled_indices]
                loss = policy_estimator.update(filtered_states, filtered_trips, filtered_advantages, filtered_actions, sess)
                print('loss={}'.format(loss))
        else:
            #TODO: filter and reshape
            action_labels = np.reshape(np.asarray(action_labels), (-1, config.num_actions))
            loss, accuracy = policy_estimator.train(states, trips, action_labels, sess)
            print('loss={}, accuracy={}'.format(loss, accuracy))
        timers_epoch = [inference_time, gpu_inference_time, env_act_time, (datetime.datetime.now() - start_timer).total_seconds()]
        timers_epoch.extend(action_timers)
        timers.append(timers_epoch)
        if config.model_to_restore is None and index_sample > 0 and index_sample % 500 == 0:
            save_path = saver.save(sess, "./rollout_model_{}_{}_{}.ckpt".format(index_sample, config.num_history_action_use, config.max_rollout_steps))
            print("Model saved in path: %s" % save_path)

    save_path = saver.save(sess, "./rollout_model.ckpt")
    print("Model saved in path: %s" % save_path)
    print('solving time = {}'.format(datetime.datetime.now() - start))
