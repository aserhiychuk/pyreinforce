import numpy as np


class TestEnv:
    def __init__(self):
        self._cur_episode = None
        self._cur_step = None
        self._global_step = 0
        self._cur_episode_max_steps = None
        self.rewards = {}

        self.seed()

    def seed(self, seed=None):
        self._np_random = np.random.RandomState()
        self._np_random.seed(seed)

    def reset(self):
        self._cur_episode_max_steps = self._np_random.randint(10, 50)

        self._cur_episode = 0 if self._cur_episode is None else self._cur_episode + 1
        self._cur_step = 0

        s = self._get_current_state()

        return s

    def step(self, a):
        self._cur_step += 1
        self._global_step += 1

        s1 = self._get_current_state()
        r = a / 100 if isinstance(a, int) else 0
        done = (self._cur_step == self._cur_episode_max_steps)
        info = None

        if self._cur_episode not in self.rewards:
            self.rewards[self._cur_episode] = 0

        self.rewards[self._cur_episode] += r

        return s1, r, done, info

    def _get_current_state(self):
        state = (self._cur_episode, self._cur_step, self._global_step)
        state = np.array(state)

        return state


class GridWorld:
#     +---+---+---+---+
#     |   |   |   | +1|
#     +---+---+---+---+
#     |   | X |   | -1|
#     +---+---+---+---+
#     |   |   |   |   |
#     +---+---+---+---+

#     +--------+--------+--------+--------+
#     | RIGHT  | RIGHT  | RIGHT  |    +1  |
#     +--------+--------+--------+--------+
#     |    UP  |    X   |  LEFT  |    -1  |
#     +--------+--------+--------+--------+
#     |    UP  |  LEFT  |  LEFT  |  DOWN  |
#     +--------+--------+--------+--------+

    def __init__(self):
        self._states = [
            [1, 1, 1, 'T'],
            [1, 0, 1, 'T'],
            [1, 1, 1,   1]
        ]
        self._rewards = [
            [0, 0, 0,  1],
            [0, 0, 0, -1],
            [0, 0, 0,  0]
        ]

        self._step_no = None

        self.seed()

    def seed(self, seed=None):
        self._np_random = np.random.RandomState()
        self._np_random.seed(seed)

    def reset(self):
        i, j = None, None

        while (i, j) == (None, None) or self._states[i][j] != 1:
            i = self._np_random.randint(0, 3)
            j = self._np_random.randint(0, 4)

        self._position = (i, j)
        self._step_no = 0

        return self._get_state()

    def step(self, action):
        actions = [action, (action - 1) % 4, (action + 1) % 4]
        action = self._np_random.choice(actions, p=[0.8, 0.1, 0.1])

        i, j = self._position

        if action == 0:
            i -= 1
        elif action == 1:
            j += 1
        elif action == 2:
            i += 1
        elif action == 3:
            j -= 1

        i = np.clip(i, 0, 2)
        j = np.clip(j, 0, 3)

        if self._states[i][j] == 0:
            i, j = self._position
        else:
            self._position = (i, j)

        reward = self._rewards[i][j]
        done = self._states[i][j] == 'T'
        self._step_no += 1

        if self._step_no == 100:
            done = True

        return self._get_state(), reward, done, None

    def _get_state(self):
        return np.array(self._position)


class LinearQBrain:
    def __init__(self, n_states, n_actions, lr, seed=None):
        self._n_states = n_states
        self._n_actions = n_actions
        self._lr = lr

        self._np_random = np.random.RandomState()
        self._np_random.seed(seed)

        self._states_one_hot = np.identity(n_states)
        self._actions_one_hot = np.identity(n_actions)
        self._w = np.ones([n_states, n_actions])

    def predict_q(self, s, **kwargs):
        if s.ndim < 2:
            s = np.expand_dims(s, axis=0)

        s = s[:, 0] * 4 + s[:, 1]
        s = self._states_one_hot[s]

        q = s @ self._w

        return q

    def train(self, s, a, t, **kwargs):
        batch_size = s.shape[0]
        batch_indices = np.arange(batch_size)
        # convert coordinates to numbers
        s = s[:, 0] * 4 + s[:, 1]
        a = np.reshape(a, -1)

        # x(S, A)
        x = np.zeros([batch_size, self._n_states, self._n_actions])
        x[batch_indices, s, a] = 1

        # q(S, A)
        q = x * self._w
        q = np.sum(q, axis=(1, 2))
        q = np.reshape(q, (batch_size, 1))

        # error term
        error = t - q
        error = np.reshape(error, (batch_size, 1, 1))

        q_grad = x * self._w
        self._w += self._lr * np.mean(error * q_grad, axis=0)
