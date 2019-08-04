import random
import numpy as np

from pyreinforce.utils import discount_rewards

class Memory(object):
    '''
    TODO Memory class
    '''
    def __init__(self, capacity, batch_size):
        self._capacity = capacity
        self._samples = []
        self._batch_size = batch_size
        self._buffer = []

        self.seed()

    def seed(self, seed=None):
        self._random = random.Random()
        self._random.seed(seed)

        return self._random

    def add(self, sample, **kwargs):
        if kwargs.get('buffer', False):
            self._buffer.append(sample)
        else:
            self._add(sample)

    def _add(self, sample):
        self._samples.append(sample)

        if len(self._samples) > self._capacity:
            self._samples.pop(0)

    def _add_all(self, samples):
        self._samples += samples

        if len(self._samples) > self._capacity:
            self._samples = self._samples[len(self._samples) - self._capacity:]

    def sample(self, **kwargs):
        batch_size = min(self._batch_size, len(self._samples))

        return self._random.sample(self._samples, batch_size)

    def _flush(self):
        self._add_all(self._buffer)

    def flush(self, gamma=None):
        if gamma is not None:
            buffer = np.array(self._buffer)
            buffer[:, 2] = discount_rewards(buffer[:, 2], gamma)

            self._buffer = buffer.tolist()

        self._flush()
        self._buffer = []


class EpisodicMemory(Memory):
    '''
    TODO EpisodicMemory class
    '''
    def __init__(self, capacity, batch_size, n_time_steps):
        super().__init__(capacity, batch_size)

        self._n_time_steps = n_time_steps

    def add(self, sample, **kwargs):
        self._buffer.append(sample)

    def sample(self, **kwargs):
        batch = super().sample()
        batch = [self._sample_time_steps(episode) for episode in batch]

        return batch

    def _sample_time_steps(self, episode):
        start = self._random.randint(0, len(episode) - self._n_time_steps)

        return episode[start : start + self._n_time_steps]

    def _flush(self):
        if len(self._buffer) < self._n_time_steps:
            return

        self._add(self._buffer)
