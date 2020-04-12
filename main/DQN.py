from typing import Tuple
from gym.wrappers.frame_stack import LazyFrames
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import random
import math
from .ExperienceReplayMemory import ExperienceReplayMemory
from torch.optim import Adam

SEED = 2020

CONV_1_FILT = 32
CONV_1_KERN = 8
CONV_1_STRI = 4
CONV_2_FILT = 64
CONV_2_KERN = 4
CONV_2_STRI = 2
CONV_3_FILT = 64
CONV_3_KERN = 3
CONV_3_STRI = 1
LINEAR_SIZE = 512

LEARNING_RATE = 0.001
BATCH_SIZE = 32
GAMMA = 0.9
EPS_END = 0.1
EPS_START = 1
EPS_DECAY = 4_000_000

COPY_EACH = 10000
SAVE_EACH = 500000
LEARN_EACH = 3


class DQN:

    def __init__(self, input_shape, output_layer_size: int, memory_capacity: int = 4):
        self.input_shape = input_shape
        self.output_layer_size = output_layer_size
        self.build_network()
        self.memory = ExperienceReplayMemory(memory_capacity)
        self.step = 0
        self.learn_step = 0
        np.random.seed(2020)
        random.seed(2020)
        torch.manual_seed(2020)

    def build_network(self):
        linear_input_size = DQN.inner_multiply(DQN.shape_after_conv(
                                DQN.shape_after_conv(
                                    DQN.shape_after_conv(self.input_shape, CONV_1_FILT, CONV_1_KERN, CONV_1_STRI),
                                    CONV_2_FILT, CONV_2_KERN, CONV_2_STRI),
                                CONV_3_FILT, CONV_3_KERN, CONV_3_STRI))
        self.online_network = nn.Sequential(
            nn.Conv2d(self.input_shape[0], CONV_1_FILT, kernel_size=CONV_1_KERN, stride=CONV_1_STRI, padding_mode='same'),
            nn.ReLU(),
            nn.Conv2d(CONV_1_FILT, CONV_2_FILT, kernel_size=CONV_2_KERN, stride=CONV_2_STRI, padding_mode='same'),
            nn.ReLU(),
            nn.Conv2d(CONV_2_FILT, CONV_3_FILT, kernel_size=CONV_3_KERN, stride=CONV_3_STRI, padding_mode='same'),
            nn.ReLU(),
            nn.modules.flatten.Flatten(),
            nn.Linear(linear_input_size, LINEAR_SIZE),
            nn.ReLU(),
            nn.Linear(LINEAR_SIZE, self.output_layer_size)
        )
        self.target_network = nn.Sequential(
            nn.Conv2d(self.input_shape[0], CONV_1_FILT, kernel_size=CONV_1_KERN, stride=CONV_1_STRI,
                      padding_mode='same'),
            nn.ReLU(),
            nn.Conv2d(CONV_1_FILT, CONV_2_FILT, kernel_size=CONV_2_KERN, stride=CONV_2_STRI, padding_mode='same'),
            nn.ReLU(),
            nn.Conv2d(CONV_2_FILT, CONV_3_FILT, kernel_size=CONV_3_KERN, stride=CONV_3_STRI, padding_mode='same'),
            nn.ReLU(),
            nn.modules.flatten.Flatten(),
            nn.Linear(linear_input_size, LINEAR_SIZE),
            nn.ReLU(),
            nn.Linear(LINEAR_SIZE, self.output_layer_size)
        )
        self.optimizer = Adam(self.online_network.parameters(), LEARNING_RATE)

    @staticmethod
    def shape_after_conv(in_shape: Tuple[int, int, int], filters: int, kernel: int, stride: int) -> Tuple[int, int, int]:
        output = int((in_shape[1] - kernel) / stride) + 1
        return filters, output, output

    @staticmethod
    def inner_multiply(tup: Tuple[int, int, int]) -> int:
        return tup[0] * tup[1] * tup[2]

    def copy_model(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save_model(self, path: str):
        torch.save(self.target_network.state_dict(), path)

    def add(self, experience):
        self.memory.push(experience)

    def run(self, state: LazyFrames, steps_done) -> int:
        epsilon_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
        if random.random() > epsilon_threshold:
            with torch.no_grad():
                prediction = self.online_network(torch.Tensor(state).squeeze().unsqueeze(0) / 255.0)
                return prediction.data.max(1)[1].numpy()[0]
        else:
            return random.randrange(self.output_layer_size)

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return

        if self.step % COPY_EACH == 0:
            self.copy_model()

        if self.step % SAVE_EACH == 0:
            self.save_model("checkpoint{0}.pth".format(self.step))

        if self.learn_step < LEARN_EACH:
            self.learn_step += 1
            return

        batch = random.sample(self.memory.memory, BATCH_SIZE)
        state, next_state, action, reward, done = map(torch.Tensor, zip(*batch))
        state = state / 255.0
        next_state = next_state / 255.0
        state.requires_grad = True

        q = self.online_network(state.squeeze())
        next_q = self.target_network(next_state.squeeze())
        a = torch.argmax(q, dim=1)
        target_q = reward + (1.0 - done) * GAMMA * next_q[np.arange(0, BATCH_SIZE), a]

        loss = F.mse_loss(q[np.arange(0, BATCH_SIZE), action.type(torch.LongTensor)], target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step = 0
