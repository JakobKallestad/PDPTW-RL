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
from tensorboardX import SummaryWriter

from sequence_encoder import encode_seq, embed_seq

from Utils import get_distance, embed_solution, objective_function
from Operators import remove_insert, remove_single_best, remove_longest_tour_deviation, remove_tour_neighbors,\
    remove_xs, remove_s, remove_m, remove_l, remove_xl, insert_first, insert_greedy, insert_beam_search, insert_tour


EPSILON = 1e-6

logs_number = 'logs_100'
writer = SummaryWriter(logs_number)


def get_config(args=None):
    parser = argparse.ArgumentParser(description="Meta optimization")
    parser.add_argument('--num_nodes', type=int, default=100, help="size of the problem for training")
    parser.add_argument('--num_episode', type=int, default=1000, help="number of training episode")
    parser.add_argument('--max_rollout_steps', type=int, default=1000, help="maximum rollout steps")
    parser.add_argument('--max_rollout_seconds', type=int, default=1000000, help="maximum rollout time in seconds")
    parser.add_argument('--sample_actions_in_rollout', type=bool, nargs='?', const=True, default=True, help="")
    parser.add_argument('--num_actions', type=int, default=29, help="dimension of action space")
    parser.add_argument('--problem_seed', type=int, default=1, help="problem generating seed")
    parser.add_argument('--input_embedded_trip_dim_2', type=int, default=11, help="")
    parser.add_argument('--num_embedded_dim_1', type=int, default=64, help="")
    parser.add_argument('--discount_factor', type=float, default=0.9, help="discount factor of policy network")
    parser.add_argument('--policy_learning_rate', type=float, default=0.001, help="learning rate of policy network")
    parser.add_argument('--hidden_layer_dim', type=int, default=64, help="dimension of hidden layer in policy network")
    parser.add_argument('--num_history_action_use', type=int, default=0, help="number of history actions used in the representation of current state")
    parser.add_argument('--step_interval', type=int, default=100)

    parser.add_argument('--model_to_restore', type=str, default=None, help="")
    parser.add_argument('--max_num_training_epsisodes', type=int, default=10000000, help="")

    parser.add_argument('--dataset_from_file', type=str, default=None, help='')
    # --model_to_restore logs_32_model_200_6.ckpt --dataset_from_file data/pdp_100/seed1234_size100_num100.pkl

    config = parser.parse_args(args)
    return config


config = get_config()



CAPACITY_MAP = {
    10: 10,  # 20,
    20: 15,  # 30,
    50: 20,  # 40,
    100: 25,  # 50
}

# Operators and logic
remove_operators = [remove_single_best, remove_longest_tour_deviation, remove_tour_neighbors,
                    remove_xs, remove_s, remove_m, remove_l, remove_xl]
insert_operators = [insert_greedy, insert_beam_search, insert_tour, insert_first]  # insert_first

len_r_op = len(remove_operators)
len_i_op = len(insert_operators)

operators = [(remove_operators[i // len_i_op], insert_operators[i % len_i_op]) for i in
             range(len_i_op - 1, len_r_op * len_i_op)]

operator_names = {
    i: f"{i:02d}" + ":    " + '   &   '.join((str(a).split()[1], str(b).split()[1]))
    for i, (a, b) in enumerate(operators)
}
operator_names[0] = "00:    remove_and_insert_single_best"



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


def construct_solution(pdp):
    return [i for i in range(1, pdp.size + 1)]


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
    return layer_2


TEST_X = tf.placeholder(tf.float32, [None, config.num_nodes, config.input_embedded_trip_dim_2])
embedded_x = embedding_net_2(TEST_X)
#embedded_x = embedding_net_attention(TEST_X)
env_observation_space_n = config.num_history_action_use * 2 + 9
action_labels_placeholder = tf.placeholder("float", [None, config.num_actions])


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
                num_outputs=config.num_actions,
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
        feed_dict = {self.states: states, TEST_X: test_x}
        return sess.run(self.action_probs, feed_dict)

    def update(self, states, test_x, advantages, actions, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.states: states, self.advantages: advantages, self.actions: actions, TEST_X:test_x}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


def env_act(pdp, solution, action):
    op = operators[action]
    new_solution, cost = remove_insert(pdp, solution, op)
    return new_solution, cost


def env_generate_state(min_distance=None, state=None, action=None, distance=None, next_distance=None, was_changed=None,
                       unseen=None, no_improvement=None, step=None):
    if state is None:
        next_state = [0.0, 0.0, 0, was_changed, unseen, no_improvement, step]
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
        next_state = [0.0, next_distance - min_distance, delta, was_changed, unseen, no_improvement, step]  # Do I need to 1-hot encode the action here?
        if config.num_history_action_use != 0:
            next_state.extend(state[(-config.num_history_action_use * 2):])
        next_state.append(delta_sign)
        next_state.append(action)
    return next_state


def get_reward(d_E, distance, next_distance, solution, next_solution, min_distance, unseen, was_changed, T):
    if next_distance < min_distance:
        next_solution = next_solution
        next_distance = next_distance
        reward = -d_E*100
    elif d_E < 0:
        next_solution = next_solution
        next_distance = next_distance
        if unseen:
            reward = -d_E*10
        else:
            reward = -d_E*5
    elif next_distance < float('inf') and unseen and random.random() < math.e ** (-d_E / T):
        next_solution = next_solution
        next_distance = next_distance
        reward = 0.01  # kind of random, but designed to be less than the other rewards above for pdp100
    else:
        next_solution = solution
        next_distance = distance
        if was_changed:
            reward = 0
        else:
            reward = -0.5  # kind of random, may have to go with something less extreme
    return next_solution, next_distance, reward



def env_step(step, state, pdp, min_distance, solution, distance, action, memory, no_improvement, T=None):
    start_timer = datetime.datetime.now()
    next_solution, next_distance = env_act(pdp, solution, action)
    unseen = str(next_solution) not in memory
    was_changed = solution != next_solution

    next_state = env_generate_state(min_distance, state, action, distance, next_distance, was_changed=int(was_changed),
                                    unseen=int(unseen), no_improvement=no_improvement, step=step)

    d_E = next_distance - distance
    next_solution, next_distance, reward = get_reward(d_E, distance, next_distance, solution, next_solution, min_distance,
                                                      unseen, was_changed, T)

    end_timer = datetime.datetime.now()
    action_timers[action * 2] += 1
    action_timers[action * 2 + 1] += (end_timer - start_timer).total_seconds()
    done = (datetime.datetime.now() - env_start_time).total_seconds() >= config.max_rollout_seconds
    return next_state, reward, done, next_solution, next_distance


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # print ([str(i.name) for i in not_initialized_vars])
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


previous_solution = None
initial_solution = None
best_solution = None
action_timers = [0.0] * (config.num_actions * 2)

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
    start = datetime.datetime.now()
    seed = config.problem_seed
    tf.set_random_seed(seed)

    Transition = collections.namedtuple("Transition", ["state", "trip", "next_distance", "action", "reward", "next_state", "done"])
    for index_sample in range(config.num_episode):

        # SA:
        T_0 = 5
        alpha = 0.996
        T = T_0

        states = []
        trips = []
        actions = []
        advantages = []
        action_labels = []
        if index_sample > 0:
            print(f"index sample: {index_sample}")

        # PDP data:
        if config.dataset_from_file is not None:
            pdp = dataset[index_sample]
        else:
            pdp = generate_problem(size=config.num_nodes)
        #pdp.initialize_close_calls()
        solution = construct_solution(pdp)
        best_solution = copy.copy(solution)
        memory = {str(solution)}  # used to check if solution has been previously encountered or not.

        embedded_trip = embed_solution(pdp, solution)
        min_distance = objective_function(pdp, solution)
        min_step = 0
        distance = min_distance

        state = env_generate_state()
        env_start_time = datetime.datetime.now()
        episode = []
        current_best_distances = []
        current_distances = []

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

            if config.sample_actions_in_rollout:
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = np.argmax(action_probs)

            end_timer = datetime.datetime.now()
            inference_time += (end_timer - start_timer).total_seconds()
            start_timer = end_timer

            next_state, reward, done, next_solution, next_distance = env_step(step, state, pdp, min_distance, solution,
                                                                              distance, action, memory, no_improvement,
                                                                              T=T)
            memory.add(str(next_solution))
            if next_distance >= distance - EPSILON:
                no_improvement += 1
            else:
                no_improvement = 0
                num_improvements += 1

            current_distances.append(distance)
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
                print('time ={}'.format('\t'.join([f"{x:.1f}" for x in temp_time_timers])))
                print('count={}'.format('\t'.join([f"{int(x)}" for x in temp_count_timers])))
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

            # Reducing temperature according to SA cooling schedule
            T *= alpha

            # Monitoring Tensorboard:
            if index_sample % 100 == 0:
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
                    operator_names[i]: prob for i, prob in enumerate(action_probs)
                }, step)


        start_timer = datetime.datetime.now()
        distances.append(min_distance)
        if objective_function(pdp, best_solution) < float('inf'):
            print('solution={}'.format(best_solution))
        else:
            print(f'invalid solution: {best_solution}')
            assert False

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

        with open(f'{logs_number}/results.txt', 'a') as file:
            file.write(str(min_distance)+'\n')

        # Warmup period
        for t, transition in enumerate(episode[:100]):
            actions.append(0)
            if transition.reward > 0:
                advantages.append(0.01)
            else:
                advantages.append(-0.01)

        #baseline_value = current_best_distances[100]

        # Search period
        for t, transition in enumerate(episode[100:]):
            actions.append(transition.action)

            discounted_future_reward = 0
            discount = 1
            for k, trans in enumerate(episode[t:]):
                discounted_future_reward += trans.reward*discount
                discount *= config.discount_factor
            advantages.append(discounted_future_reward)


        states = np.reshape(np.asarray(states), (-1, env_observation_space_n)).astype("float32")
        trips = np.reshape(np.asarray(trips), (-1, config.num_nodes, config.input_embedded_trip_dim_2)).astype("float32")
        actions = np.reshape(np.asarray(actions), (-1))
        advantages = np.reshape(np.asarray(advantages), (-1)).astype("float32")

        print('num_random_actions={}'.format(num_random_actions))
        print('actions={}'.format(actions[:100]).replace('\n', ''))
        print('advantages={}'.format(advantages[:100]).replace('\n', ''))
        if config.model_to_restore is None:
            loss = policy_estimator.update(states, trips, advantages, actions, sess)
            print('loss={}'.format(loss))
        print(f"inference_time={inference_time}", f"gpu_inference_time={gpu_inference_time}",
              f"env_act_time={env_act_time}", f"post_episode_time={(datetime.datetime.now() - start_timer).total_seconds()}",
              sep='\t')
        if config.model_to_restore is None and index_sample > 0 and index_sample % 100 == 0:
            save_path = saver.save(sess, f"./trained_models/{logs_number}_model_{index_sample}_{config.num_history_action_use}.ckpt")
            print("Model saved in path: %s" % save_path)

    print("DONE!")

