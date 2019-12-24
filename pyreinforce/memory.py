import random
import numpy as np

from pyreinforce.utils import discount_rewards

class Memory(object):
    """Experience replay memory that stores recent experiences.

    Experience (aka transition) is a tuple of (`state`, `action`, `reward`,
    `next state`, `terminal flag`).
    """

    def __init__(self, capacity, batch_size):
        """
        Parameters
        ----------
        capacity : int
            Maximum number of experiences to store.
        batch_size : int
            Number of sampled experiences.
        """
        self._capacity = capacity
        self._samples = []
        self._batch_size = batch_size
        self._buffer = []

        self.seed()

    def seed(self, seed=None):
        """Seed the random number generator.

        Parameters
        ----------
        seed : int, optional
            Seed for random number generator.
        """
        self._random = random.Random()
        self._random.seed(seed)

        return self._random

    def add(self, sample, **kwargs):
        """Add given experience to the memory.

        Parameters
        ----------
        sample : sequence or obj
            Experience, typically a tuple of (`s`, `a`, `r`, `s1`, `s1_mask`).

        Keyword arguments
        -----------------
        buffer : bool, optional
            If `True` an experience will be placed in the buffer,
            and added to the memory otherwise.
        """
        if kwargs.get('buffer', False):
            self._buffer.append(sample)
        else:
            self._add(sample)

    def _add(self, sample):
        """Add given experience to the memory.

        Parameters
        ----------
        sample : sequence or obj
            Experience, typically a tuple of (`s`, `a`, `r`, `s1`, `s1_mask`).
        """
        self._samples.append(sample)

        if len(self._samples) > self._capacity:
            self._samples.pop(0)

    def _add_all(self, samples):
        """Add a list of experiences to the memory.

        Parameters
        ----------
        samples : list
            List of experiences.
        """
        self._samples += samples

        if len(self._samples) > self._capacity:
            self._samples = self._samples[len(self._samples) - self._capacity:]

    def sample(self, **kwargs):
        """Sample a batch of experiences.

        Returns
        -------
        list
            Batch of `self._batch_size` experiences.
        """
        batch_size = min(self._batch_size, len(self._samples))

        return self._random.sample(self._samples, batch_size)

    def _flush(self):
        """Append all pending experiences to the memory."""
        self._add_all(self._buffer)

    def flush(self, gamma=None):
        """Append all pending experiences to the memory,
        optionally discount rewards first.

        Parameters
        ----------
        gamma : float, optional
            Discount factor, must be between 0 and 1.
        """
        if gamma is not None:
            buffer = np.array(self._buffer)
            buffer[:, 2] = discount_rewards(buffer[:, 2], gamma)

            self._buffer = buffer.tolist()

        self._flush()
        self._buffer = []


class EpisodicMemory(Memory):
    """Experience replay memory that stores recent episodes."""

    def __init__(self, capacity, batch_size, n_time_steps):
        """
        Parameters
        ----------
        capacity : int
            Maximum number of episodes to store.
        batch_size : int
            Number of sampled sequences of experiences.
        n_time_steps : int
            Number of time steps in each sequence.
        """
        super().__init__(capacity, batch_size)

        self._n_time_steps = n_time_steps

    def add(self, sample, **kwargs):
        """Add given experience to the buffer.

        Parameters
        ----------
        sample : sequence or obj
            Experience, typically a tuple of (`s`, `a`, `r`, `s1`, `s1_mask`).
        """
        self._buffer.append(sample)

    def sample(self, **kwargs):
        """Sample a batch of sequences of experiences.

        Returns
        -------
        list
            Batch of `self._batch_size` sequences, where each sequence
            consists of `self._n_time_steps` experiences.
        """
        batch = super().sample()
        batch = [self._sample_time_steps(episode) for episode in batch]

        return batch

    def _sample_time_steps(self, episode):
        """Sample a sequence of `self._n_time_steps` consecutive experiences
        from an episode.

        Parameters
        ----------
        episode
            List of all experiences in the episode.

        Returns
        -------
        list
            Sequence of `self._n_time_steps` consecutive experiences.
        """
        start = self._random.randint(0, len(episode) - self._n_time_steps)

        return episode[start : start + self._n_time_steps]

    def _flush(self):
        """Append a pending episode to the memory."""
        if len(self._buffer) < self._n_time_steps:
            return

        self._add(self._buffer)
