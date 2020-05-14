# PyReinforce
**PyReinforce** is a collection of algorithms that can be used to solve a variety of different reinforcement learning problems.

| Breakout | Pong | Space Invaders |
|:-:|:-:|:-:|
| [![Breakout](assets/Breakout_160x210.gif?raw=true)](examples/Atari.ipynb) | [![Pong](assets/Pong_160x210.gif?raw=true)](examples/Atari.ipynb) | [![Space Invaders](assets/SpaceInvaders_160x210.gif?raw=true)](examples/Atari.ipynb) |

| Cart Pole | Lunar Lander | Pendulum |
|:-:|:-:|:-:|
| [![Cart Pole](assets/CartPole_160x160.gif?raw=true)](examples/MonteCarlo.ipynb) | [![Lunar Lander](assets/LunarLander_160x160.gif?raw=true)](examples/TemporalDifference.ipynb) | [![Pendulum](assets/Pendulum_160x160.gif?raw=true)](examples/DDPG.ipynb) |

## Basics
This is how [OpenAI Gym](https://gym.openai.com/) describes the reinforcement learning process:
>There are two basic concepts in reinforcement learning: the environment (namely, the outside world) and the agent (namely, >the algorithm you are writing). The agent sends actions to the environment, and the environment replies with observations and >rewards (that is, a score).

PyReinforce is fully compatible with [OpenAI Gym](https://gym.openai.com/). In fact, it will work with **any** environment as long as it implements these methods:
* `reset()`
* `step(action)`

Your part is to implement a `Brain`: a neural network that agents use to decide which actions to pick for given states. Agents get better over time by performing training steps on their brains. See [examples](examples) for more details.

## Installation
Install PyReinforce from PyPI:
```bash
pip install PyReinforce
```
or from source:
```bash
git clone https://github.com/aserhiychuk/pyreinforce.git
cd pyreinforce
pip install -e .
```

## Examples
* [Monte Carlo](examples/MonteCarlo.ipynb)
* [Temporal Difference](examples/TemporalDifference.ipynb)
* [Policy Gradient](examples/PolicyGradient.ipynb)
* [Deep Deterministic Policy Gradient](examples/DDPG.ipynb)
* [Deep Q-Learning](examples/Atari.ipynb)

In order to run the examples you need to install dependencies:
```bash
pip install -r examples/requirements.txt
```
