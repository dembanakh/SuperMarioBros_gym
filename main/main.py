from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import matplotlib.pyplot as plt
from torch import FloatTensor, LongTensor, cat, zeros
from .DQN import DQN, SEED
from .wrappers import wrapper

# CONSTANTS
TRIALS = 5000
FAILURE_REWARD = -1
MEMORY_CAPACITY = 10000
###

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = wrapper(env)
env.seed(SEED)

state_size = (4, 84, 84)
actions_size = env.action_space.n
model: DQN = DQN(state_size, actions_size, MEMORY_CAPACITY)
trial_results = []
steps_done = 0


def run_trial(i):
    global steps_done
    state = env.reset()
    total_reward = 0
    step = 0
    while True:
        # env.render()
        action = model.run(state, steps_done)
        next_state, reward, done, _ = env.step(action)
        steps_done += 1

        model.add((state, next_state, action, reward, done))
        model.learn()

        total_reward += reward
        state = next_state
        step += 1

        if done:
            print("Trial {0} finished after {1} steps"
                  .format(i, step))
            trial_results.append(step)
            plot_durations()
            break


def plot_results():
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Trial')
    plt.ylabel('Steps')
    plt.plot(np.array(trial_results))

    plt.pause(0.001)


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = FloatTensor(trial_results)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = cat((zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


for _ in range(TRIALS):
    run_trial(_)

env.close()

plot_results()
plt.ioff()
plt.show()
