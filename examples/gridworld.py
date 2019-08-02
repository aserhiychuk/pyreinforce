from time import sleep
import itertools

import numpy as np
from skimage.transform import rescale
from skimage import img_as_ubyte 
import warnings

import gym
from gym import spaces
from gym.utils import seeding


class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

   
class GridWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }
    reward_range = (-1, 1)

    def __init__(self, size=5, max_episode_steps=50, partial=False):
        super().__init__()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)

        self._size_x = size
        self._size_y = size
        self._max_episode_steps = max_episode_steps
        self._partial = partial

        self._objects = []
        self._cur_step = None
        self._viewer = None

        self.seed()

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):
        self._objects = []
        hero = gameOb(self._new_position(), 1, 1, 2, None, 'hero')
        self._objects.append(hero)
        bug = gameOb(self._new_position(), 1, 1, 1, 1, 'goal')
        self._objects.append(bug)
        hole = gameOb(self._new_position(), 1, 1, 0, -1, 'fire')
        self._objects.append(hole)
        bug2 = gameOb(self._new_position(), 1, 1, 1, 1, 'goal')
        self._objects.append(bug2)
        hole2 = gameOb(self._new_position(), 1, 1, 0, -1, 'fire')
        self._objects.append(hole2)
        bug3 = gameOb(self._new_position(), 1, 1, 1, 1, 'goal')
        self._objects.append(bug3)
        bug4 = gameOb(self._new_position(), 1, 1, 1, 1, 'goal')
        self._objects.append(bug4)

        width = self.observation_space.shape[1]
        height = self.observation_space.shape[0]
        state = self._to_image(width, height)

        self._cur_step = 0

        return state

    def step(self, action):
        self._move_char(action)
        reward = self._check_goal()

        width = self.observation_space.shape[1]
        height = self.observation_space.shape[0]
        state = self._to_image(width, height)

        self._cur_step += 1

        done = bool(self._cur_step >= self._max_episode_steps)

        return state, reward, done, None

    def _move_char(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self._objects[0]

        if direction == 0 and hero.y >= 1:
            hero.y -= 1

        if direction == 1 and hero.y <= self._size_y - 2:
            hero.y += 1

        if direction == 2 and hero.x >= 1:
            hero.x -= 1

        if direction == 3 and hero.x <= self._size_x - 2:
            hero.x += 1

        self._objects[0] = hero

    def _new_position(self):
        iterables = [ range(self._size_x), range(self._size_y)]
        points = []

        for t in itertools.product(*iterables):
            points.append(t)

        currentPositions = []

        for objectA in self._objects:
            if (objectA.x,objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))

        for pos in currentPositions:
            points.remove(pos)

        location = self._np_random.choice(range(len(points)), replace=False)

        return points[location]

    def _check_goal(self):
        others = []

        for obj in self._objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)

        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self._objects.remove(other)

                if other.reward == 1:
                    self._objects.append(gameOb(self._new_position(), 1, 1, 1, 1, 'goal'))
                else: 
                    self._objects.append(gameOb(self._new_position(), 1, 1, 0, -1, 'fire'))

                return other.reward

        return 0

    def _to_image(self, width, height=None, mode=None):
        a = np.ones([self._size_y + 2, self._size_x + 2, 3])
        a[1:-1, 1:-1, :] = 0
        hero = None

        for item in self._objects:
            a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, item.channel] = item.intensity

            if item.name == 'hero':
                hero = item

        if self._partial:
            visible_area = a[hero.y:hero.y + 3, hero.x:hero.x + 3, :]

            if mode == 'human':
                black = np.all(a == [0, 0, 0], axis=-1)
                visible_black = np.all(visible_area == [0, 0, 0], axis=-1)

                a[black] = 0.5
                a *= 0.25
                visible_area *= 4
                visible_area[visible_black] = 0
            else:
                a = visible_area

        scale_width = width / a.shape[1]
        scale_height = height / a.shape[0] if height else scale_width

        a = rescale(a, (scale_height, scale_width), order=0, multichannel=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = img_as_ubyte(a)

        return a

    def render(self, mode='human', close=False):
        if mode == 'human':
            if self._viewer is None:
                from gym.envs.classic_control import rendering
                self._viewer = rendering.SimpleImageViewer()

            s = self._to_image(400, mode='human')

            self._viewer.imshow(s)

            sleep(1)
        elif mode == 'rgb_array':
            width = self.observation_space.shape[1]
            height = self.observation_space.shape[0]

            return self._to_image(width, height)
        else:
            super().render(mode=mode)

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
