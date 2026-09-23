"""Running reward scaling for on-policy fine-tuning.

Rewards are divided by the standard deviation of a rolling discounted sum of
the rewards, without re-centering, so that the value targets stay on a scale the
freshly initialized critic can fit. Ported from `reference/dppo`'s
`util/reward_scaling.py`, which follows the PPG reward normalizer
(https://github.com/openai/phasic-policy-gradient) and section 9.3 of
https://arxiv.org/pdf/1811.02553.
"""

import numpy as np


class RunningMeanStd:
    """Welford-style running moments over a stream of batches.

    Attributes:
        shape: Unbatched shape of the tracked statistic.
        epsilon: Initial count, so that the first update cannot divide by zero.
    """

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        """Fold a batch of samples into the running moments."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / (total_count - 1)
        self.count = total_count


def backward_discounted_sum(prevret, reward, first, gamma):
    """Discounted sum of the rewards seen so far, per environment.

    Args:
        prevret: (num_envs,) running sum carried over from the previous call.
        reward: (num_envs, num_steps) rewards, oldest first.
        first: (num_envs, num_steps) 1 where an episode begins.
        gamma: Discount of the rolling sum.

    Returns:
        (num_envs, num_steps) running sums.
    """
    assert first.ndim == 2
    _, num_steps = reward.shape
    ret = np.zeros_like(reward)
    for step in range(num_steps):
        prevret = ret[:, step] = reward[:, step] + (
            1 - first[:, step]
        ) * gamma * prevret
    return ret


class RunningRewardScaler:
    """Scale rewards by the running std of their rolling discounted sum.

    The statistic is over the *time-reversed* returns, because the forward
    returns are not known yet while the rollout is being collected.

    Attributes:
        num_envs: Number of parallel environments.
        clip_reward: Absolute bound applied after scaling.
        gamma: Discount of the rolling sum; independent of the PPO discount.
        epsilon: Variance floor.
    """

    def __init__(self, num_envs, clip_reward=10.0, gamma=0.99, epsilon=1e-8):
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_reward = clip_reward
        self.ret = np.zeros(num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def __call__(self, reward, first):
        """Update the statistics with a rollout and return the scaled rewards.

        Args:
            reward: (num_envs, num_steps) rewards, oldest first.
            first: (num_envs, num_steps) 1 where an episode begins.

        Returns:
            (num_envs, num_steps) scaled rewards.
        """
        rets = backward_discounted_sum(
            prevret=self.ret, reward=reward, first=first, gamma=self.gamma
        )
        self.ret = rets[:, -1]
        self.ret_rms.update(rets.reshape(-1))
        return self.transform(reward)

    def transform(self, reward):
        """Apply the current scaling without updating the statistics."""
        return np.clip(
            reward / self.scale, -self.clip_reward, self.clip_reward
        )

    @property
    def scale(self):
        """Current divisor, reported alongside the training metrics."""
        return np.sqrt(self.ret_rms.var + self.epsilon)
