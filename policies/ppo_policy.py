from infrastructure.base_class import BasePolicy
from itertools import chain
import torch
from torch import nn, distributions
from infrastructure.utils import normalize
import infrastructure.pytorch_util as ptu
import numpy as np


class MLPPolicy(BasePolicy, nn.Module):
    def __init__(self, ac_dim, mean_net, logits_na, clip_grad_norm,
                 optimizer_spec, baseline_optim_spec=None, baseline_network=None,
                 discrete=False, nn_baseline=False, training=True, **kwargs):
        super(MLPPolicy, self).__init__(**kwargs)
        self.ac_dim = ac_dim
        self.discrete = discrete
        self.training = training
        self.clip_grad_norm = clip_grad_norm
        self.nn_baseline = nn_baseline
        self.mean_net = mean_net
        self.logits_na = logits_na
        self.baseline = baseline_network
        self.optimizer_spec = optimizer_spec
        self.baseline_optim_spec = baseline_optim_spec
        if self.discrete:
            self.logits_na.to(ptu.device)
            parameters = self.logits_na.parameters()
        else:
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device))
            self.logstd.to(ptu.device)
            parameters = chain([self.logstd], self.mean_net.parameters())
        self.optimizer, self.lr_schedule = ptu.build_optim(self.optimizer_spec, parameters)
        if nn_baseline:
            self.baseline.to(ptu.device)
            self.baseline_loss = nn.MSELoss()
            self.baseline_optimizer, self.baseline_lr_schedule = ptu.build_optim(self.baseline_optim_spec, self.baseline.parameters())

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def forward(self, observation, return_log_prob=False):
        if self.discrete:
            dist = distributions.Categorical(logits=self.logits_na(observation))
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(batch_mean, scale_tril=batch_scale_tril)
            dist = action_distribution
        if return_log_prob:
            action = dist.rsample()
            return action, dist.log_prob(action)
        else:
            return dist

    def get_action(self, obs):
        self.eval()
        observation = ptu.from_numpy(obs.astype(np.float32))
        return ptu.to_numpy(self(observation).sample())

    def get_log_prob(self, obs, acs):
        obs = ptu.from_numpy(obs)
        acs = ptu.from_numpy(acs)
        return self(obs).log_prob(acs).detach()

    def update(self, obs, action, **kwargs):
        raise NotImplementedError


class MLPPolicyPPO(MLPPolicy):
    def __init__(self, ac_dim, mean_net, logits_na,
                 clip_grad_norm, optimizer_spec, ppo_eps,
                 discrete=False, training=True, **kwargs):
        super(MLPPolicyPPO, self).__init__(ac_dim, mean_net, logits_na, clip_grad_norm, optimizer_spec,
                                          None, None, discrete, False, training, **kwargs)
        self.ppo_eps = ppo_eps

    def update(self, obs, action, adv_n=None, q_vals=None):
        self.train()
        observation = ptu.from_numpy(obs)
        action = ptu.from_numpy(action)
        if isinstance(adv_n, np.ndarray):
            adv_n = ptu.from_numpy(adv_n)
        log_prob = self(observation).log_prob(action)
        if q_vals is None:
            loss = - torch.min(log_prob*adv_n, torch.clamp(1-self.ppo_eps, 1+self.ppo_eps)*adv_n).mean()
        else:
            loss = - (log_prob*adv_n).mean()
        # print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        if self.discrete:
            nn.utils.clip_grad_norm_(self.logits_na.parameters(), self.clip_grad_norm)
        else:
            nn.utils.clip_grad_norm_(self.mean_net.parameters(), self.clip_grad_norm)
            nn.utils.clip_grad_norm_(self.logstd, self.clip_grad_norm/10)
        self.optimizer.step()
        if self.optimizer_spec[2]:
            self.lr_schedule.step()
        return {'Training Loss': loss.item()}
