import numpy as np


def discount_rewards(rewards, gamma):
    """Compute discounted rewards.

    Parameters
    ----------
    rewards : list
        List of rewards obtained from the environment.
    gamma : float
        Discount factor, must be between 0 and 1.

    Returns
    -------
    list
        List of discounted rewards.
    """
    result = np.empty_like(rewards, dtype=np.float32)
    g = 0

    for i in reversed(range(len(rewards))):
        g = rewards[i] + gamma * g
        result[i] = g

    return result
