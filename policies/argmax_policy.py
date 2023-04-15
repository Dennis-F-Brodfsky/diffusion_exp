from infrastructure.base_class import BasePolicy
from infrastructure.utils import softmax, sample_discrete
import numpy as np


class ArgmaxPolicy(BasePolicy):
    def __init__(self, critic, use_boltzmann=False):
        self.critic = critic
        self.use_boltzmann = use_boltzmann

    def update(self, obs, action, **kwargs):
        pass

    def get_action(self, obs):
        q_values = self.critic.qa_values(obs)
        if not self.use_boltzmann:
            return np.argmax(q_values, axis=1)
        else:
            distribution = softmax(q_values)
            return sample_discrete(distribution)

    def save(self, filepath: str):
        pass

