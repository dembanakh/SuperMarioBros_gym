from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from torch import Tensor, no_grad, load
from .DQN import DQN, SEED
from .wrappers import wrapper
from sys import argv

# CONSTANTS
TRIALS = 10

PATH = argv[1] if len(argv) > 1 else "checkpoint0.pth"
###

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = wrapper(env)
env.seed(SEED)

state_size = (4, 84, 84)
actions_size = env.action_space.n
model: DQN = DQN(state_size, actions_size)
model.target_network.load_state_dict(load(PATH))


def choose_action(state):
    with no_grad():
        prediction = model.target_network(Tensor(state).squeeze().unsqueeze(0))
        return prediction.data.max(1)[1].numpy()[0]


def run_trial(i):
    state = env.reset()
    total_reward = 0
    step = 0
    while True:
        env.render()
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)

        total_reward += reward
        state = next_state
        step += 1

        if done:
            print("Trial {0} finished after {1} steps"
                  .format(i, step))
            break
    input()


for _ in range(TRIALS):
    run_trial(_)

env.close()
