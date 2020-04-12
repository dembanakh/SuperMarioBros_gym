from collections import deque
import random


class ExperienceReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, sample):
        if len(self.memory) == self.memory.maxlen:
            self.memory.popleft()
        self.memory.append(sample)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
