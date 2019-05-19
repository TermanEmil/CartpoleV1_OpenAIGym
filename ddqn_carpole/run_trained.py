import gym
import numpy as np
from tensorflow.keras.models import load_model


# Constants
c_env_name = "CartPole-v1"
c_max_nb_of_steps = 2000000

c_trained_model_file_path = './trained_models/trained_v0.h5'

# Globals
g_state_size = None
g_action_size = None

g_env = None
g_model = None


def get_action(state):
    q_values = g_model.predict(state)[0]
    return np.argmax(q_values)


def reshape_state(state):
    return np.reshape(state, [1, g_state_size])


if __name__ == '__main__':
    g_env = gym.make(c_env_name)
    g_env._max_episode_steps = c_max_nb_of_steps

    g_state_size = g_env.observation_space.shape[0]
    g_action_size = g_env.action_space.n

    g_model = load_model(c_trained_model_file_path)

    while True:
        state = reshape_state(g_env.reset())
        done = 0

        while not done:
            g_env.render()

            action = get_action(state)
            next_state, reward, done, info = g_env.step(action)
            next_state = reshape_state(next_state)

            state = next_state
