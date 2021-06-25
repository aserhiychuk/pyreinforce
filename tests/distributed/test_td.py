import unittest

from pyreinforce.acting import EpsGreedyPolicy, DecayingEpsGreedyPolicy
from pyreinforce.distributed import ExperienceBuffer
from pyreinforce.distributed.td import AsyncTdAgent, TdWorker

from tests import TestEnv as DummyEnv, GridWorld, assert_grid_world_policy
from tests.distributed import DummyQBrain, DummyConn, DummySharedWeights, AsyncLinearQBrain


def _create_dummy_env(worker_no=None):
    return DummyEnv()

def _create_dummy_brain(worker_no=None):
    return DummyQBrain(3, 7)

def _create_grid_world_env(worker_no=None):
    return GridWorld()

def _create_grid_world_brain(worker_no=None):
    n_states = 12
    n_actions = 4
    lr = 0.025
    seed = 123

    return AsyncLinearQBrain(n_states, n_actions, lr, seed)


class AsyncTdAgentTest(unittest.TestCase):
    def test_run(self):
        n_episodes = 10
        acting = EpsGreedyPolicy(0)
        experience_buffer = ExperienceBuffer()
        gamma = 0.99
        train_freq = 5
        validation_freq = None
        validation_episodes = None
        converter = None
        callback = None
        n_workers = 4

        agent = AsyncTdAgent(n_episodes, _create_dummy_env, _create_dummy_brain,
                             acting, experience_buffer, gamma, train_freq,
                             validation_freq, validation_episodes,
                             converter, callback, n_workers)
        rewards, _ = agent.run()

        self.assertSequenceEqual(range(n_workers), sorted(rewards.keys()))

        for _, worker_rewards in rewards.items():
            self.assertEqual(n_episodes, len(worker_rewards))

    def test_run_learning(self):
        start_eps = 1
        end_eps = 0
        eps_decay = 1
        acting = DecayingEpsGreedyPolicy(start_eps, end_eps, eps_decay)
        experience_buffer = ExperienceBuffer()

        n_episodes = 500
        gamma = 0.99
        train_freq = 3
        validation_freq = None
        validation_episodes = None
        converter = None
        callback = None
        n_workers = 8

        agent = AsyncTdAgent(n_episodes, _create_grid_world_env, _create_grid_world_brain,
                             acting, experience_buffer, gamma, train_freq,
                             validation_freq, validation_episodes,
                             converter, callback, n_workers)
        rewards, _ = agent.run()

        self.assertSequenceEqual(range(n_workers), sorted(rewards.keys()))

        for _, worker_rewards in rewards.items():
            self.assertEqual(n_episodes, len(worker_rewards))

        assert_grid_world_policy(agent._brain.predict_q)


class TdWorkerTest(unittest.TestCase):
    def setUp(self):
        worker_no = 3
        conn_to_parent = DummyConn(worker_no)
        brain = DummyQBrain(3, 9)
        weights_metadata = [(w.shape, w.dtype, w.nbytes) for w in brain.get_weights()]
        shared_weights = DummySharedWeights(weights_metadata)
        barrier = None
        self._n_episodes = 10
        env = DummyEnv()
        acting = EpsGreedyPolicy(0)
        experience_buffer = ExperienceBuffer()
        gamma = 0.99
        train_freq = 5
        converter = None
        callback = None

        self._worker = TdWorker(worker_no, conn_to_parent, shared_weights, barrier,
                                self._n_episodes, env, brain, acting, experience_buffer,
                                gamma, train_freq, converter, callback)

    def test_run(self):
        rewards, _ = self._worker.run()

        self.assertEqual(self._n_episodes, len(rewards))


if __name__ == '__main__':
    unittest.main()
