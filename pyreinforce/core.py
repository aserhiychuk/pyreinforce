import time

import numpy as np


class Agent(object):
    """Base class for all Reinforcement Learning agents."""

    def __init__(self):
        super().__init__()
#         self._logger = logging.getLogger('{}.{}'.format(__name__, type(self).__name__))

    def seed(self, seed=None):
        """Seed the random number generator.

        Parameters
        ----------
        seed : int, optional
            Seed for random number generator.

        Note
        ----
        Base method does nothing.
        """
        pass

    def run(self):
        """Train the agent.

        Note
        ----
        Base method does nothing.
        """
        pass


class SimpleAgent(Agent):
    """Abstract agent that implements a typical RL flow."""

    def __init__(self, n_episodes, env, validation_freq=None, validation_episodes=None,
                 converter=None, callback=None):
        """
        Parameters
        ----------
        n_episodes : int
            Number of episodes to train the agent for.
        env : obj
            Environment
        validation_freq : int, optional
            Specifies how many episodes to run before a new validation run is performed.
        validation_episodes : int, optional
            Number of episodes in each validation run.
        converter : Converter, optional
            If specified, allows to pre/post process state, action, or experience.
        callback : callable, optional
            If specified, is called after each episode
            with the following parameters:

            cur_episode : int
                Episode number.
            reward : float
                Episode cumulative reward.
            **kwargs
                Additional keyword arguments.
        """
        super().__init__()
        self._n_episodes = n_episodes
        self._env = env
        self._converter = converter
        self._validation_freq = validation_freq
        self._validation_episodes = validation_episodes
        self._callback = callback

        self._global_step = 0

    def run(self):
        """Train the agent by running `self._n_episodes` episodes.

        Returns
        -------
        list
            Validation rewards if `validation_freq` is specified,
            training rewards otherwise.
        obj
            Some useful training statistics.
        """
        training_rewards = []
        validation_rewards = []
        stats = []

        for cur_episode in range(self._n_episodes):
            episode_start = time.perf_counter()

            self._before_episode(cur_episode)
            reward, n_episode_steps = self._run_episode(cur_episode)
            self._after_episode()

            episode_stop = time.perf_counter()
            stats.append(episode_stop - episode_start)

            training_rewards.append(reward)

            if callable(self._callback):
                kwargs = {
                    'n_episode_steps': n_episode_steps,
                    'global_step': self._global_step,
                    'rewards': training_rewards,
                    'n_episodes': self._n_episodes
                }
                self._callback(cur_episode, reward, **kwargs)

            if self._validation_freq and (cur_episode + 1) % self._validation_freq == 0:
                self._before_validation()

                rewards = []

                for _ in range(self._validation_episodes):
                    reward = self._validate_episode()
                    rewards.append(reward)

                self._after_validation(rewards)

                validation_rewards.append(rewards)

        rewards = validation_rewards if self._validation_freq else training_rewards

        stats = np.array(stats)
        stats = stats.min(), stats.max(), stats.mean(), stats.std()

        return rewards, stats

    def _run_episode(self, cur_episode):
        """Run a single episode.

        Parameters
        ----------
        cur_episode : int
            Episode number.

        Returns
        -------
        float
            Episode cumulative reward.
        int
            Number of episode steps.
        """
        cur_step = 0
        reward = 0
        done = False
        s = self._reset()

        if self._converter:
            s = self._converter.convert_state(s)

        while not done:
            a = self._act(s, cur_step=cur_step, cur_episode=cur_episode,
                          n_episodes=self._n_episodes, global_step=self._global_step)

            experience = (s, a)

            if self._converter:
                a = self._converter.convert_action(a)

            s1, r, done, info = self._step(a)

            if self._converter:
                s1 = self._converter.convert_state(s1, info)

            experience += (r, s1, done)

            if self._converter:
                experience = self._converter.convert_experience(experience, info)

            self._observe(experience)

            s = s1

            reward += r
            cur_step += 1
            self._global_step += 1

        return reward, cur_step

    def _validate_episode(self):
        """Run a single validation episode.

        Returns
        -------
        float
            Episode cumulative reward.
        """
        cur_step = 0
        reward = 0
        done = False
        s = self._reset()

        if self._converter:
            s = self._converter.convert_state(s)

        while not done:
            a = self._act(s, validation=True, cur_step=cur_step)

            if self._converter:
                a = self._converter.convert_action(a)

            s1, r, done, info = self._step(a)

            if self._converter:
                s1 = self._converter.convert_state(s1, info)

            s = s1

            reward += r
            cur_step += 1

        return reward

    def _reset(self):
        """Bring the environment to its initial state.

        Returns
        -------
        obj
            Initial state.
        """
        s = self._env.reset()

        return s

    def _step(self, a):
        """Take step in the environment by performing action `a` on it.

        Parameters
        ----------
        a int or obj
            Action to be performed.

        Returns
        -------
        obj
            Next state after performing action `a`.
        float
            Reward achieved by action `a`.
        bool
            Flag that indicates if the episode has terminated.
        dict or obj
            Diagnostic information useful for debugging.
        """
        s1, r, done, info = self._env.step(a)

        return s1, r, done, info

    def _before_episode(self, cur_episode=0):
        """Called before an episode starts.

        Parameters
        ----------
        cur_episode : int, optional
            Current episode number.
        """
        pass

    def _act(self, s, validation=False, **kwargs):
        """Return action given state `s`.

        Parameters
        ----------
        s
            Current state.
        validation : bool, optional
            Indicator that is True during validation.
        **kwargs
            cur_step : int, optional
                Current step within episode.
            cur_episode : int, optional
                Current episode.
            n_episodes : int, optional
                Total number of episodes.
            global_step : int, optional
                Global step across all episodes.

        Returns
        -------
        int or obj
            Action to be performed.

        Note
        ----
        Must be implemented by subclasses. Base method
        raises `NotImplementedError` exception.
        """
        raise NotImplementedError

    def _observe(self, experience):
        """Called after taking a step in the environment.

        Subclasses can use this method to save current `experience`
        in replay memory and/or perform a training step.

        Parameters
        ----------
        experience : tuple
            Tuple that contains information about the current step:
            s
                State before performing action.
            a : int or obj
                Action that was performed.
            r : float
                Reward achieve by the action.
            s1
                State after performing action.
            terminal_flag : bool
                True if `s1` is a terminal state, False otherwise.
        """
        pass

    def _after_episode(self):
        """Called after an episode ends."""
        pass

    def _before_validation(self):
        pass

    def _after_validation(self, rewards):
        pass
