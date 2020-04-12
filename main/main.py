from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import time
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
rewards = list()
steps_done = 0
start_time = time.time()


def run_trial(i):
    global steps_done
    state = env.reset()
    total_reward = 0
    step = 0
    while True:
        env.render()
        action = model.run(state, steps_done)
        next_state, reward, done, info = env.step(action)
        steps_done += 1

        model.add((state, next_state, action, reward, done))
        model.learn()

        total_reward += reward
        state = next_state
        step += 1

        if done or info['flag_get']:
            print("Trial {0} finished after {1} steps"
                  .format(i, step))
            rewards.append(total_reward / step)
            if i % 100 == 0:
                print('Episode {e} - +'
                      'Frame {f} - +'
                      'Frames/sec {fs} - +'
                      'Mean Reward {r}'.format(e=i,
                                               f=model.step,
                                               fs=np.round((model.step - steps_done) / (time.time() - start_time)),
                                               r=np.mean(rewards[-100:])))
            break


def plot_results():
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Trial')
    plt.ylabel('Steps')
    plt.plot(np.array(rewards))

    plt.pause(0.001)


for _ in range(TRIALS):
    run_trial(_)

env.close()

plot_results()
plt.ioff()
plt.show()

np.save('rewards.npy', rewards)
