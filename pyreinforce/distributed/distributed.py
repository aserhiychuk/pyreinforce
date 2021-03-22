import random
import logging
import time
from uuid import uuid4
from threading import Thread
import multiprocessing as mp
from multiprocessing import connection, Process, Pipe, Barrier, Value
from multiprocessing.managers import SharedMemoryManager

import numpy as np

from pyreinforce.core import Agent, SimpleAgent, Callback


class DistributedAgent(Agent):
    def __init__(self, n_episodes, env, brain, train_freq, validation_freq=None,
                 validation_episodes=None, converter=None, callback=None,
                 n_workers=None):
        self._logger = logging.getLogger(f'{__name__}.{type(self).__name__}')

        self._n_episodes = n_episodes
        self._env = env
        self._brain = brain
        self._train_freq = train_freq
        self._converter = converter
        self._callback = callback

        if n_workers is None:
            self._n_workers = mp.cpu_count()
        else:
            self._n_workers = n_workers

        self._validation_freq = validation_freq

        if validation_episodes:
            # Distribute validation load between workers as equally as possible.
            # Each worker will grab value that corresponds to its number.
            if validation_episodes < n_workers:
                # n_workers = 4, validation_episodes = 3
                # [1, 1, 1, 0]
                self._validation_episodes = [1] * validation_episodes + [0] * (n_workers - validation_episodes)
            elif n_workers < validation_episodes:
                # n_workers = 4, validation_episodes = 5
                # [2, 1, 1, 1]
                #
                # n_workers = 4, validation_episodes = 15
                # [4, 4, 4, 3]
                self._validation_episodes = [validation_episodes // n_workers + 1] * (validation_episodes % n_workers)
                self._validation_episodes += [validation_episodes // n_workers] * (n_workers - validation_episodes % n_workers)
            else:
                # n_workers = 4, validation_episodes = 4
                # [1, 1, 1, 1]
                self._validation_episodes = [1] * n_workers

            assert n_workers == len(self._validation_episodes)
            assert validation_episodes == sum(self._validation_episodes)
        else:
            self._validation_episodes = None

        self._rewards = None
        self._info = None

        self._grads_queue_sizes = []

    def run(self):
        self._rewards = {}
        self._info = {
            'workers': {}
        }

        barrier = Barrier(self._n_workers)

        brain = self._brain()
        weights = brain.get_weights()
        weights_metadata = [(w.shape, w.dtype, w.nbytes) for w in weights]
        weights_size = sum([w.nbytes for w in weights])

        with SharedMemoryManager() as smm:
            shared_memory = smm.SharedMemory(size=self._n_workers * weights_size)
            self._logger.info(f'allocated shared memory of size {shared_memory.size}. total workers weights size {self._n_workers * weights_size}')
            shared_weights = SharedWeights(weights_metadata, shared_memory)

            worker_processes = []
            conns_to_workers = []

            for worker_no in range(self._n_workers):
                conn_to_parent, conn_to_worker = Pipe()

                shared_weights.write(worker_no, weights)

                worker_process = Process(name=f'worker-{worker_no}', target=self._worker_process, args=(worker_no, conn_to_parent, shared_weights, barrier))
                worker_process.start()

                # worker will be the only one who has reference to parent
                conn_to_parent.close()

                worker_processes.append(worker_process)
                # TODO _workers_listener mutates this list
                conns_to_workers.append(conn_to_worker)

            self._logger.info('workers have been spun up')

            # persisting the brain after spinning up sub processes
            self._brain = brain

            workers_listener = Thread(name='workers-listener', daemon=True,
                                      target=self._workers_listener, args=(conns_to_workers, shared_weights))
            workers_listener.start()

            self._logger.info('started workers listener thread')

            # TODO consider using futures to join sub processes
            for worker_no, worker_process in enumerate(worker_processes):
                worker_process.join()

            self._logger.info('workers finished')

            workers_listener.join()

            self._logger.info('workers listener thread stopped')

        self._info.update({
            'grads_queue_sizes': self._grads_queue_sizes
        })

        if self._validation_freq:
            rewards = [np.array(worker_rewards) for _, worker_rewards in self._rewards.items()]
            rewards = np.concatenate(rewards, axis=1)
            rewards = rewards.tolist()
            self._rewards = rewards

        return self._rewards, self._info

    def _worker_process(self, worker_no, conn_to_parent, shared_weights, barrier):
        self._logger.info(f'[worker_no {worker_no}] starting...')

        _env = self._env(worker_no) if callable(self._env) else self._env
        _brain = self._brain(worker_no) if callable(self._brain) else self._brain

        worker = self._create_worker(worker_no, conn_to_parent, shared_weights, barrier, _env, _brain)
        rewards, info = worker.run()

        conn_to_parent.send({
            'type'     : 'result',
            'ref_id'   : uuid4(),
            'worker_no': worker_no,
            'payload'  : (rewards, info)
        })
        conn_to_parent.close()

        self._logger.info(f'[worker_no {worker_no}] done')

    def _create_worker(self, worker_no, conn_to_parent, shared_weights, barrier, env, brain, *args, **kwargs):
        raise NotImplementedError

    def _workers_listener(self, conns, shared_weights):
        while conns:
            # TODO check performance of connectin.wait()
            ready_conns = connection.wait(conns)

            self._grads_queue_sizes.append(len(ready_conns))

            # shuffle to increase randomness
            random.shuffle(ready_conns)

            for conn in ready_conns:
                try:
                    msg = conn.recv()
                except EOFError:
                    conns.remove(conn)
                    conn.close()
                else:
                    if msg['type'] == 'gradients':
                        ref_id = msg['ref_id']
                        worker_no = msg['worker_no']

                        # read and apply the gradients from shared memory
                        grads = shared_weights.read(worker_no)
                        self._brain.apply_gradients(grads)

                        # update worker's shared memory with new weights
                        weights = self._brain.get_weights()
                        shared_weights.write(worker_no, weights)

                        # let worker know new weights are available 
                        conn.send({
                            'type'  : 'weights',
                            'ref_id': ref_id
                        })
                    elif msg['type'] == 'result':
                        ref_id = msg['ref_id']
                        worker_no = msg['worker_no']
                        rewards, info = msg['payload']

                        self._rewards[worker_no] = rewards
                        self._info['workers'][worker_no] = info

                        self._logger.info(f'received result from worker_no {worker_no}')
                    else:
                        raise RuntimeError(f'Unsupported message type: {msg["type"]}')

        self._logger.info('exiting workers listener thread')


class WorkerAgent(SimpleAgent):
    def __init__(self, worker_no, conn_to_parent, shared_weights, barrier,
                 n_episodes, env, brain, train_freq, validation_freq=None, validation_episodes=None,
                 converter=None, callback=None):

        if isinstance(callback, Callback):
            callback = WorkerCallback(worker_no, callback)

        super().__init__(n_episodes, env, validation_freq, validation_episodes,
                         converter, callback)

        self._worker_no = worker_no
        self._conn_to_parent = conn_to_parent
        self._shared_weights = shared_weights
        self._barrier = barrier
        self._brain = brain
        weights = self._shared_weights.read(self._worker_no)
        self._brain.set_weights(weights)
        self._train_freq = train_freq

        self._experience_buffer = []

        self._latencies = []

    def run(self):
        rewards, stats = super().run()

        info = {
            'stats'    : stats,
            'latencies': self._latencies
        }

        return rewards, info

    def _observe(self, experience):
        self._experience_buffer.append(experience)

        if (len(self._experience_buffer) == self._train_freq) or (experience[-1] is True):
            self._train(self._experience_buffer)

            self._experience_buffer = []

    def _train(self, batch):
        batch_size = len(batch)

        s = [s for s, _, _, _, _ in batch]
        s = np.array(s)
        s = np.reshape(s, (batch_size, -1))

        a = [a for _, a, _, _, _ in batch]
        a = np.array(a)
        a = np.reshape(a, (batch_size, 1))

        r = [r for _, _, r, _, _ in batch]
        r = np.array(r)
        r = np.reshape(r, (batch_size, 1))

        s1 = [s1 for _, _, _, s1, _ in batch]
        s1 = np.array(s1)
        s1 = np.reshape(s1, (batch_size, -1))

        s1_mask = [1 - done for _, _, _, _, done in batch]
        s1_mask = np.array(s1_mask)
        s1_mask = np.reshape(s1_mask, (batch_size, 1))

        batch = s, a, r, s1, s1_mask

        grads = self._compute_grads(batch)

        # persist gradients in shared memory
        self._shared_weights.write(self._worker_no, grads)

        # let the learner know gradients are available 
        req_ref_id = uuid4()
        t1 = time.perf_counter()
        self._conn_to_parent.send({
            'type'     : 'gradients',
            'ref_id'   : req_ref_id,
            'worker_no': self._worker_no
        })
        # wait until gradients are applied and updated weights become available
        # TODO use timeout and raise error if no response received
        res = self._conn_to_parent.recv()
        t2 = time.perf_counter()
        self._latencies.append((t1, t2))
        assert res['type'] == 'weights'
        assert req_ref_id == res['ref_id']
        # TODO assert response status (i.e. OK? not OK?)

        # update worker's weights
        weights = self._shared_weights.read(self._worker_no)
        self._brain.set_weights(weights)

    def _compute_grads(self, batch):
        raise NotImplementedError

    def _before_validation(self):
        super()._before_validation()

        self._barrier.wait()

        latest_weights = self._shared_weights.read_latest()
        self._brain.set_weights(latest_weights)


class WorkerCallback(Callback):
    def __init__(self, worker_no, callback):
        self._worker_no = worker_no
        self._callback = callback

    def on_before_run(self, **kwargs):
        self._callback.on_before_run(worker_no=self._worker_no, **kwargs)

    def on_after_run(self, **kwargs):
        self._callback.on_after_run(worker_no=self._worker_no, **kwargs)

    def on_state_change(self, s, **kwargs):
        self._callback.on_state_change(s, worker_no=self._worker_no, **kwargs)

    def on_before_episode(self, episode_no, **kwargs):
        self._callback.on_before_episode(episode_no, worker_no=self._worker_no, **kwargs)

    def on_after_episode(self, episode_no, reward, **kwargs):
        self._callback.on_after_episode(episode_no, reward, worker_no=self._worker_no, **kwargs)

    def on_before_validation(self, **kwargs):
        self._callback.on_before_validation(worker_no=self._worker_no, **kwargs)

    def on_after_validation(self, rewards, **kwargs):
        self._callback.on_after_validation(rewards, worker_no=self._worker_no, **kwargs)


class SharedWeights:
    def __init__(self, metadata, shared_memory):
        self._metadata = metadata
        self._weights_size = sum([nbytes for _, _, nbytes in metadata])
        assert shared_memory.size >= self._weights_size, f'Shared memory size too small. Shared memory: {shared_memory.size}, weights: {self._weights_size}'
        self._shared_memory = shared_memory

        self._latest_worker_no = Value('i', -1)

    def write(self, worker_no, data):
        offset = worker_no * self._weights_size

        for weights in data:
            shared_weights = np.ndarray(weights.shape, weights.dtype, self._shared_memory.buf, offset)
            shared_weights[:] = weights[:]

            offset += weights.nbytes

        self._latest_worker_no.value = worker_no

    def read(self, worker_no):
        weights = []
        offset = worker_no * self._weights_size

        for shape, dtype, nbytes in self._metadata:
            shared_weights = np.ndarray(shape, dtype, self._shared_memory.buf, offset)
            weights.append(shared_weights)

            offset += nbytes

        return weights

    def read_latest(self):
        latest_worker_no = self._latest_worker_no.value

        if latest_worker_no == -1:
            raise ValueError('Weights are not initialized')

        return self.read(latest_worker_no)
