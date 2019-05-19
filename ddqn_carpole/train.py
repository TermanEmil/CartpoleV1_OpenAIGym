import random
import gym
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

# Constants
c_env_name = "CartPole-v1"

c_max_nb_of_steps = 2000

c_discount_rate = 0.99
c_learning_rate = 0.001

c_memory_size = 20000
c_batch_size = 64

# Required memory to start training
c_mem_len_train_start = 1000

c_exploration_max = 1.0
c_exploration_min = 0.01
c_exploration_decay = 0.999

# Globals
g_state_size = None
g_action_size = None

g_env = None
g_memory = deque(maxlen=c_memory_size)

g_model = None
g_target_model = None

g_epsilon = c_exploration_max


def build_model():
    model = Sequential()

    model.add(Dense(24, activation='relu', input_dim=g_state_size, kernel_initializer='he_uniform'))
    model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(g_action_size, kernel_initializer='he_uniform'))

    model.compile(Adam(lr=c_learning_rate), 'mse')
    return model


def update_target_model():
    g_target_model.set_weights(g_model.get_weights())


def get_action(state):
    if np.random.rand() < g_epsilon:
        return g_env.action_space.sample()
    else:
        q_values = g_model.predict(state)[0]
        return np.argmax(q_values)


def append_memory(state, action, reward, next_state, done):
    global g_epsilon

    g_memory.append((state, action, reward, next_state, done))

    if g_epsilon > c_exploration_min:
        g_epsilon = max(g_epsilon * c_exploration_decay, c_exploration_min)


def can_train():
    mem_len = len(g_memory)
    return mem_len >= c_mem_len_train_start and mem_len >= c_batch_size


def train_model():
    if not can_train():
        return

    mini_batch = random.sample(g_memory, c_batch_size)

    update_input = np.zeros((c_batch_size, g_state_size))
    update_target = np.zeros((c_batch_size, g_state_size))
    action, reward, done = [], [], []

    for i in range(c_batch_size):
        update_input[i] = mini_batch[i][0]
        action.append(mini_batch[i][1])
        reward.append(mini_batch[i][2])
        update_target[i] = mini_batch[i][3]
        done.append(mini_batch[i][4])

    target = g_model.predict(update_input)
    target_next = g_model.predict(update_target)
    target_val = g_target_model.predict(update_target)

    for i in range(c_batch_size):
        if done[i]:
            target[i][action[i]] = reward[i]
        else:
            # The key point of Double DQN:
            #  Selection of action is from model
            #  Update is from target model
            a = np.argmax(target_next[i])
            target[i][action[i]] = reward[i] + c_discount_rate * target_val[i][a]

    g_model.fit(
        update_input,
        target,
        batch_size=c_batch_size,
        epochs=1,
        verbose=0)


def reshape_state(state):
    return np.reshape(state, [1, g_state_size])


def run_episode():
    done = False
    score = 0
    state = reshape_state(g_env.reset())

    while not done:
        g_env.render()

        action = get_action(state)
        next_state, reward, done, info = g_env.step(action)
        next_state = reshape_state(next_state)

        if done and score < c_max_nb_of_steps - 1:
            reward = -100

        append_memory(state, action, reward, next_state, done)
        train_model()
        score += 1
        state = next_state

    return score


if __name__ == '__main__':
    g_env = gym.make(c_env_name)
    g_env._max_episode_steps = c_max_nb_of_steps

    g_state_size = g_env.observation_space.shape[0]
    g_action_size = g_env.action_space.n

    g_model = build_model()
    g_target_model = build_model()

    scores = []

    for episode in range(10000):
        update_target_model()

        score = run_episode()
        scores.append(score)
        score_mean = float(np.mean(scores[-min(10, len(scores)):]))

        print(
            "episode: %3d, score: %4d, epsilon: %.4f, mean: %4.2f" %
            (episode, score, g_epsilon, score_mean))

        # if the mean of scores of last 10 episode is bigger than 80%
        # stop training
        if score_mean > 0.8 * c_max_nb_of_steps:
            g_model.save('trained_models/trained_v0.h5')
            exit(0)
