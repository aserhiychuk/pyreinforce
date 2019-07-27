import random


class Memory(object):
    '''
    TODO Memory class
    '''
    def __init__(self, capacity):
        self._capacity = capacity
        self._samples = []
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

    def sample(self, batch_size, **kwargs):
        batch_size = min(batch_size, len(self._samples))

        return self._random.sample(self._samples, batch_size)

    def _flush(self):
        self._add_all(self._buffer)

    def flush(self, preprocess_buffer=None):
        if callable(preprocess_buffer):
            self._buffer = preprocess_buffer(self._buffer)

        self._flush()
        self._buffer = []
