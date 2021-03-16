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

logs_number = 'logs_31'
writer = SummaryWriter(logs_number)

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
    parser.add_argument('--train_operators', type=str2bool, nargs='?', const=True, default=False, help="")

    parser.add_argument('--num_training_points', type=int, default=100, help="size of the problem for training")
    parser.add_argument('--num_test_points', type=int, default=100, help="size of the problem for testing")
    parser.add_argument('--num_episode', type=int, default=40000, help="number of training episode")
    parser.add_argument('--max_num_rows', type=int, default=2000000, help="")
    parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
    parser.add_argument('--max_rollout_steps', type=int, default=20000, help="maximum rollout steps")
    parser.add_argument('--max_rollout_seconds', type=int, default=1000, help="maximum rollout time in seconds")
    parser.add_argument('--use_rl_loss', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--epsilon_greedy', type=float, default=0.05, help="")
    parser.add_argument('--sample_actions_in_rollout', type=str2bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--num_active_learning_iterations', type=int, default=1, help="")
    parser.add_argument('--max_no_improvement', type=int, default=6, help="")
    parser.add_argument('--debug_mode', type=str2bool, nargs='?', const=True, default=False, help="")
    parser.add_argument('--debug_steps', type=int, default=1, help="")
    parser.add_argument('--num_actions', type=int, default=24, help="dimension of action space")
    parser.add_argument('--problem_seed', type=int, default=1, help="problem generating seed")
    parser.add_argument('--input_embedded_trip_dim_2', type=int, default=11, help="")
    parser.add_argument('--num_embedded_dim_1', type=int, default=64, help="")
    parser.add_argument('--num_embedded_dim_2', type=int, default=64, help="dim")
    parser.add_argument('--discount_factor', type=float, default=1.0, help="discount factor of policy network")
    parser.add_argument('--policy_learning_rate', type=float, default=0.001, help="learning rate of policy network")
    parser.add_argument('--hidden_layer_dim', type=int, default=64, help="dimension of hidden layer in policy network")
    parser.add_argument('--num_history_action_use', type=int, default=0, help="number of history actions used in the representation of current state")
    parser.add_argument('--step_interval', type=int, default=100)

    # './rollout_model_1850.ckpt'
    parser.add_argument('--model_to_restore', type=str, default=None, help="")
    parser.add_argument('--max_num_training_epsisodes', type=int, default=10000000, help="")

    parser.add_argument('--dataset_from_file', type=str, default=None, help='')
    # --model_to_restore logs_30_model_500_6.ckpt --dataset_from_file data/pdp_100/seed1234_size100_num100.pkl

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
insert_operators = [insert_greedy, insert_beam_search, insert_tour]  # insert_first,

len_r_op = len(remove_operators)
len_i_op = len(insert_operators)
n_operators = len_r_op * len_i_op

operator_names = {
    i: f"{i:02d}" + ":    " + '   &   '.join((str(remove_operators[i // len_i_op]).split()[1], str(insert_operators[i % len_i_op]).split()[1])) for i in range(config.num_actions)
}
operator_names[0] = "00:    Perturb"
operator_names[1] = "01:    remove_and_insert_single_best"
operator_names[2] = "02:    remove_and_insert_single_best"


class PDP:
    def __init__(self, size, n_calls, locations, capacities, calls, dist_matrix):
        self.size = size
        self.n_calls = n_calls
        self.locations = locations
        self.capacities = capacities
        self.calls = calls
        self.dist_matrix = dist_matrix
        #self.initialize_close_calls()

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


if config.dataset_from_file is not None:
    dataset = load_pdp_from_file(config.dataset_from_file)


def improve_solution_by_action(pdp, solution, action):
    if action == 0:  # Perturb
        #n = random.randint(1, 7)
        remove_op = remove_xs  # remove_operators[n]
        op = (remove_op, insert_first)
    else:  # Intensify
        remove_op = remove_operators[action // len_i_op]
        insert_op = insert_operators[action % len_i_op]
        op = (remove_op, insert_op)
    new_solution, cost = remove_insert(pdp, solution, op)
    return new_solution


is_training = tf.placeholder(tf.bool)
keep_prob = tf.placeholder(tf.float32)


def construct_solution(pdp):
    return [i for i in range(1, pdp.size + 1)]  # Try to perturb solution in a meaningful way!


def get_num_points(config):
    if config.model_to_restore is None:
        return config.num_training_points
    else:
        return config.num_test_points


ATTENTION_ROLLOUT, LSTM_ROLLOUT = True, False #False, True


def embedding_net_2(input_):
    with tf.variable_scope("embedding_net"):
        x = embed_seq(input_seq=input_, from_=config.input_embedded_trip_dim_2, to_=config.num_embedded_dim_1, is_training=True,
                      BN=True, initializer=tf.contrib.layers.xavier_initializer())

        layer_attention = encode_seq(input_seq=x, input_dim=config.num_embedded_dim_1, num_stacks=1, num_heads=8,
                                     num_neurons=64, is_training=True, dropout_rate=0.1)
        layer_2 = tf.reduce_sum(layer_attention, axis=1)
    return layer_2


def embedding_net_attention(input_):
    with tf.variable_scope("attention_embedding"):
        x = embed_seq(input_seq=input_, from_=config.num_embedded_dim, to_=128, is_training=True, BN=True, initializer=tf.contrib.layers.xavier_initializer())
        layer_attention = encode_seq(input_seq=x, input_dim=128, num_stacks=3, num_heads=16, num_neurons=512, is_training=True, dropout_rate=0.1)
        layer_attention = tf.reshape(layer_attention, [-1, config.max_trips_per_solution * 128])
        layer_2 = tf.contrib.layers.fully_connected(layer_attention, config.num_embedded_dim, activation_fn=tf.nn.relu)
        layer_2 = tf.nn.dropout(layer_2, keep_prob)
    return layer_2


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
            full_states = tf.concat([self.states, embedded_x], axis=1)

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


previous_solution = None
initial_solution = None
best_solution = None


def env_act(pdp, solution, action):
    new_solution = improve_solution_by_action(pdp, solution, action)
    cost = objective_function(pdp, new_solution)
    if cost == float('inf'):
        print('Invalid solution!')
    return new_solution, cost


action_timers = [0.0] * (config.num_actions * 2)


def env_generate_state(min_distance=None, state=None, action=None, distance=None, next_distance=None):
    if state is None or action == 0:
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


def env_step(step, state, pdp, min_distance, solution, distance, action, record_time=True, T=None):
    start_timer = datetime.datetime.now()
    new_solution, next_distance = env_act(pdp, solution, action)

    # Using SA for acceptance criteria
    d_E = next_distance - distance
    if d_E < 0:
        solution = new_solution
    elif next_distance < float('inf') and random.random() < math.e ** (-d_E / T):
        solution = new_solution
    else:
        pass
        # next_distance = distance

    next_state = env_generate_state(min_distance, state, action, distance, next_distance)
    reward = distance - next_distance
    end_timer = datetime.datetime.now()
    if record_time:
        action_timers[action * 2] += 1
        action_timers[action * 2 + 1] += (end_timer - start_timer).total_seconds()
    done = (datetime.datetime.now() - env_start_time).total_seconds() >= config.max_rollout_seconds
    return next_state, reward, done, solution, next_distance


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print ([str(i.name) for i in not_initialized_vars])
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


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

        # SA:
        T_0 = 5
        alpha = 0.98#0.993
        T = T_0

        states = []
        trips = []
        actions = []
        advantages = []
        action_labels = []
        if index_sample > 0 and index_sample % config.debug_steps == 0:
            print(f"index sample: {index_sample}")

        # PDP data:
        if config.dataset_from_file is not None:
            pdp = dataset[index_sample]
        else:
            pdp = generate_problem(size=config.num_training_points)
        pdp.initialize_close_calls()
        solution = construct_solution(pdp)
        best_solution = copy.copy(solution)

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

            states.append(state)
            trips.append(embedded_trip)

            if config.model_to_restore is "NOT PERTURBING" and no_improvement >= config.max_no_improvement:  #(config.model_to_restore is not None and should_restart(min_distance, distance, no_improvement)) or no_improvement >= config.max_no_improvement:
                action = 0
                no_improvement = 0
            else:
                if config.model_to_restore is "NOT EP GREEDY" and np.random.uniform() < config.epsilon_greedy:
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

            next_state, reward, done, next_solution, next_distance = env_step(step, state, pdp, min_distance, solution, distance, action, T=T)
            if next_distance >= distance - EPSILON:
                no_improvement += 1
            else:
                no_improvement = 0
                num_improvements += 1

            current_distances.append(distance)
            start_distances.append(start_distance)
            if action == 0:
                start_distance = next_distance
            current_best_distances.append(min_distance)
            if next_distance < min_distance - EPSILON:
                min_distance = next_distance
                min_step = step
                best_solution = copy.copy(next_solution)
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
                state=state, trip=copy.copy(embedded_trip), next_distance=next_distance,
                action=action, reward=reward, next_state=next_state, done=done))
            state = next_state
            solution = next_solution
            embedded_trip = embed_solution(pdp, solution)
            distance = next_distance
            end_timer = datetime.datetime.now()
            env_act_time += (end_timer - start_timer).total_seconds()
            start_timer = end_timer

            # SA
            T *= alpha

            # Monitoring Tensorboard:
            if index_sample % 10 == 0:
                writer.add_scalars(f"cost_{index_sample}", {
                    "incumbent": next_distance,
                    "best_cost": min_distance
                }, step)

                total_count = sum(action_timers[::2])
                total_time = sum(action_timers[1::2])
                writer.add_scalars(f'action_count_{index_sample}', {
                    operator_names[i]: x / total_count for i, x in enumerate(action_timers[::2])
                }, step)

                writer.add_scalars(f'action_timer_{index_sample}', {
                    operator_names[i]: x / total_time for i, x in enumerate(action_timers[1::2])
                }, step)

                writer.add_scalars(f"action_prob_{index_sample}", {
                    operator_names[i+1]: prob for i, prob in enumerate(action_probs)
                }, step)




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

        temp_timers = np.asarray(action_timers)
        temp_count_timers = temp_timers[::2]
        temp_time_timers = temp_timers[1::2]
        total_count = sum(temp_count_timers)
        total_time = sum(temp_time_timers)


        writer.add_scalars('action_count', {
            operator_names[i]: x/total_count for i, x in enumerate(temp_count_timers)
        }, index_sample)
        writer.add_scalars('action_timer', {
            operator_names[i]: x/total_time for i, x in enumerate(temp_time_timers)
        }, index_sample)

        # reset action_timers for each instance
        action_timers = [0.0] * (config.num_actions * 2)

        with open(f'{logs_number}_results.txt', 'a') as file:
            file.write(str(min_distance)+'\n')

        future_best_distances = [0.0] * len(episode)
        future_best_distances[-1] = episode[-1].next_distance
        step = len(episode) - 2
        while step >= 0:
            if episode[step].action != 0:
                future_best_distances[step] = future_best_distances[step + 1] * config.discount_factor
            else:
                future_best_distances[step] = current_distances[step]
            step = step - 1

        historical_baseline = 0
        for t, transition in enumerate(episode):
            # total_return = sum(config.discount_factor**i * future_transition.reward for i, future_transition in enumerate(episode[t:]))
            # if historical_baseline is None:
            #     if transition.action == 0:
            #         #TODO: dynamic updating of historical baseline, and state definition
            #         historical_baseline = -current_best_distances[t]
            #         # historical_baseline = 1/(current_best_distances[t] - 10)
            #     actions.append(0)
            #     advantages.append(0)
            #     continue
            # if transition.action == 0:
            #     historical_baseline = -current_distances[t]
            if transition.action > 0:
                #total_return = transition.reward
                if transition.reward < EPSILON:
                    total_return = -1.0
                else:
                    total_return = 1.0
                #     total_return = min(transition.reward, 2.0)
                # total_return = start_distances[t] - future_best_distances[t]
                # total_return = min(total_return, 1.0)
                # total_return = max(total_return, -1.0)
                ##total_return = -future_best_distances[t]  # this was the default one, jakob
                # total_return = 1/(future_best_distances[t] - 10)
            else:
                # if transition.state[-1] != 0 and transition.state[-2] < 1e-6:
                #     # if future_best_distances[t] < current_best_distances[t] - 1e-6:
                #     total_return = 1.0
                # else:
                #     total_return = -1.0
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
        trips = np.reshape(np.asarray(trips), (-1, config.num_training_points, config.input_embedded_trip_dim_2)).astype("float32")
        actions = np.reshape(np.asarray(actions), (-1))
        advantages = np.reshape(np.asarray(advantages), (-1)).astype("float32")

        print('num_random_actions={}'.format(num_random_actions))
        print('actions={}'.format(actions[:100]).replace('\n', ''))
        print('advantages={}'.format(advantages[:100]).replace('\n', ''))
        if config.model_to_restore is None and index_sample <= config.max_num_training_epsisodes:
            filtered_states = []
            filtered_trips = []
            filtered_advantages = []
            filtered_actions = []
            end = 0  # len(actions)+5  # 0
            for action_index in range(len(actions)):
                if actions[action_index] > 0:
                    filtered_states.append(states[action_index])
                    filtered_trips.append(trips[action_index])
                    filtered_advantages.append(advantages[action_index])
                    filtered_actions.append(actions[action_index] - 1)
                else:
                    num_bad_steps = config.max_no_improvement
                    end = max(end, len(filtered_states) - num_bad_steps)
                    filtered_states = filtered_states[:end]
                    filtered_trips = filtered_trips[:end]
                    filtered_advantages = filtered_advantages[:end]
                    filtered_actions = filtered_actions[:end]
            if end == 0:
                end = len(filtered_states)

            filtered_states = filtered_states[:end]
            filtered_trips = filtered_trips[:end]
            filtered_advantages = filtered_advantages[:end]
            filtered_actions = filtered_actions[:end]
            num_states = len(filtered_states)
            if num_states > config.batch_size:
                downsampled_indices = np.random.choice(range(num_states), config.batch_size, replace=False)
                filtered_states = np.asarray(filtered_states)[downsampled_indices]
                filtered_trips = np.asarray(filtered_trips)[downsampled_indices]
                filtered_advantages = np.asarray(filtered_advantages)[downsampled_indices]
                filtered_actions = np.asarray(filtered_actions)[downsampled_indices]
            loss = policy_estimator.update(filtered_states, filtered_trips, filtered_advantages, filtered_actions, sess)
            print('loss={}'.format(loss))
        timers_epoch = [inference_time, gpu_inference_time, env_act_time, (datetime.datetime.now() - start_timer).total_seconds()]
        timers_epoch.extend(action_timers)
        timers.append(timers_epoch)
        if config.model_to_restore is None and index_sample > 0 and index_sample % 100 == 0:
            save_path = saver.save(sess, f"./{logs_number}_model_{index_sample}_{config.num_history_action_use}.ckpt")
            print("Model saved in path: %s" % save_path)

    save_path = saver.save(sess, "./rollout_model.ckpt")
    print("Model saved in path: %s" % save_path)
    print('solving time = {}'.format(datetime.datetime.now() - start))
