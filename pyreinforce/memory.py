import random


class Memory(object):
    '''
    TODO Memory class
    '''
    def __init__(self, capacity):
        self._capacity = capacity
        self._samples = []

    def add(self, sample):
        if isinstance(sample, list):
            self._add_all(sample)
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

    def sample(self, n):
        n = min(n, len(self._samples))

        return random.sample(self._samples, n)
