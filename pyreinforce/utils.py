import numpy as np


def discount_rewards(rewards, gamma):
    result = np.empty_like(rewards, dtype=np.float32)
    g = 0

    for i in reversed(range(len(rewards))):
        g = rewards[i] + gamma * g
        result[i] = g

    return result
