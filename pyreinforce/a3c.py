import os
import time
import uuid
from threading import Thread, Lock
from multiprocessing import Process, Pipe

import numpy as np

from pyreinforce.core import Agent, SimpleAgent


class A3cAgent(Agent):
    '''
    TODO A3C Agent class
    '''
    def __init__(self, n_episodes, envs, brain, acting, gamma, replay_batch_size):
        super().__init__()

        self._n_episodes = n_episodes
        self._envs = envs
        self._brain = brain
        self._acting = acting
        self._gamma = gamma

        self._replay_memory_lock = Lock()
        self._replay_memory = []
        self._replay_batch_size = replay_batch_size

        self._result = {}
        self._result_lock = Lock()

    def run(self):
        processes = []

        for env in self._envs:
            parent_conn, child_conn = Pipe()

            process = Process(target=self._worker, args=(self._n_episodes, env, self._acting, child_conn))
            process.start()

            self._result[process.pid] = {
                'rewards': None,
                'stats': {
                    'general': None,
                    'letancy': {
                        'experience': []
                    }
                }
            }

            thread = Thread(name='subprocess-connection-{}'.format(process.pid), daemon=True, 
                            target=self._subprocess_conn, args=(parent_conn,))
            thread.start()

            processes.append(process)

        for process in processes:
            process.join()

        return self._result

    def _worker(self, n_episodes, env, acting, child_conn):
        worker = A3cWorker(n_episodes, env, acting, child_conn)
        worker.run()

    def _subprocess_conn(self, conn):
        while True:
            try:
                req = conn.recv()
            except EOFError:
                break

            if req['type'] == 'experience':
                t1, pid, experience = req['payload']
                t2 = time.perf_counter()

                with self._result_lock:
                    self._result[pid]['stats']['letancy']['experience'].append(t2 - t1)

                self._observe(experience)
            elif req['type'] == 'policy_req':
                t1, pid, ref_id, s = req['payload']

                p = self._predict_policy(s)
                conn.send((ref_id, p))
            elif req['type'] == 'result':
                t1, pid, rewards, stats = req['payload']

                with self._result_lock:
                    self._result[pid]['rewards'] = rewards
                    self._result[pid]['stats']['general'] = stats

    def _observe(self, experience):
        batch = None

        with self._replay_memory_lock:
            self._replay_memory.append(experience)

            if len(self._replay_memory) >= self._replay_batch_size:
                batch = self._replay_memory
                self._replay_memory = []

        if batch is not None:
            self._train(batch)

    def _train(self, batch):
        batch = np.array(batch)
        s = np.stack(batch[:, 0])
        a = np.vstack(batch[:, 1])
        r = np.vstack(batch[:, 2])
        s1 = np.stack(batch[:, 3])
        s1_mask = np.vstack(batch[:, 4])

        v = self._predict_value(s1)
        v *= s1_mask
        r += self._gamma * v

        self._brain.train(s, a, r)

    def _predict_value(self, states):
        v = self._brain.predict_value(states)
        assert not np.isnan(v).any(), 'value contains nan: {}'.format(v)
        assert not np.isinf(v).any(), 'value contains inf: {}'.format(v)

        return v

    def _predict_policy(self, states):
        p = self._brain.predict_policy(states)
        assert not np.isnan(p).any(), 'policy contains nan: {}'.format(p)
        assert not np.isinf(p).any(), 'policy contains inf: {}'.format(p)

        return p


class A3cWorker(SimpleAgent):
    '''
    TODO A3C Worker class
    '''
    def __init__(self, n_episodes, env, acting, conn):
        super().__init__(n_episodes, env)
        self._acting = acting
        self._pid = os.getpid()
        self._conn = conn

    def run(self):
        rewards, stats = super().run()

        self._conn.send({
            'type': 'result',
            'payload': (time.perf_counter(), self._pid, rewards, stats)
        })

        return rewards, stats

    def _act(self, s, **kwargs):
        req_ref_id = uuid.uuid4()
        req = {
            'type': 'policy_req',
            'payload': (time.perf_counter(), self._pid, req_ref_id, s)
        }
        self._conn.send(req)

        res_ref_id, probs = self._conn.recv()
        assert req_ref_id == res_ref_id, 'ref ids don\'t match. in ref_id: {}, out ref_id: {}'.format(req_ref_id, res_ref_id)

        a = self._acting.act(probs)

        return a 

    def _observe(self, experience):
        msg = {
            'type': 'experience',
            'payload': (time.perf_counter(), self._pid, experience)
        }
        self._conn.send(msg)
