from infrastructure.base_class import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
from infrastructure import pytorch_util as ptu


class DQNCritic(BaseCritic):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.double_q = hparams['double_q']
        self.grad_norm_clipping = hparams['max_norm_clipping']
        self.gamma = hparams['gamma']
        self.target_update_rate = hparams['target_update_rate']
        self.q_net_spec = hparams['q_net_spec']
        self.q_net = hparams['q_func']()
        self.q_net_target = hparams['q_func']()
        self.clipped_q = hparams['clipped_q']
        self.loss = nn.SmoothL1Loss()  # AKA Huber loss
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.q_net_target.eval()
        self.parameters = [self.q_net.parameters()]
        if self.clipped_q:
            self.q2_net = hparams['q2_func']()
            self.q2_net_target = hparams['q2_func']()
            self.q2_net.to(ptu.device)
            self.q2_net_target.to(ptu.device)
            self.q2_net_target.eval()
            self.parameters.append(self.q2_net.parameters())
        self.q_net_optimizer = self.q_net_spec[0](self.parameters, **self.q_net_spec[1])
        self.q_net_scheduler = optim.lr_scheduler.LambdaLR(self.q_net_optimizer, self.q_net_spec[2])

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        self.q_net.train()
        if self.clipped_q:
            self.q2_net.train()
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        if self.clipped_q:
            qa_t_values = self.q_net(ob_no)
            qa2_t_values = self.q2_net(ob_no)
            qa_tp1_values = torch.min(self.q_net_target(next_ob_no), self.q2_net_target(next_ob_no))
            q2_t_values = torch.gather(qa2_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
        else:
            qa_t_values = self.q_net(ob_no)
            qa_tp1_values = self.q_net_target(next_ob_no)
            q2_t_values = None
        # feature
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
        if self.double_q:
            qa_t1_action = (self.q_net(next_ob_no) if not self.clipped_q else torch.min(self.q_net(next_ob_no), self.q2_net(next_ob_no))).argmax(dim=1).unsqueeze(-1)
            q_tp1 = qa_tp1_values.gather(1, qa_t1_action).squeeze(1)
        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)
        # target / prediction
        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach()
        loss = self.loss(q_t_values, target)
        if self.clipped_q:
            loss += self.loss(q2_t_values, target)
        self.q_net_optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.parameters, self.grad_norm_clipping)
        self.q_net_optimizer.step()
        if self.q_net_spec[2]:
            self.q_net_scheduler.step()
        return {'Training Loss': ptu.to_numpy(loss)}

    def update_target_network(self):
        self.soft_update(self.q_net, self.q_net_target, self.target_update_rate)
        if self.clipped_q:
            self.soft_update(self.q2_net, self.q2_net_target, self.target_update_rate)

    def qa_values(self, obs, **kwargs):
        self.q_net.eval()
        obs = ptu.from_numpy(obs)
        if self.clipped_q:
            self.q2_net.eval()
            qa_values = torch.min(self.q_net(obs), self.q2_net(obs))
        else:
            qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)

    @staticmethod
    def soft_update(net, target_net, target_update_rate):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data * (1 - target_update_rate) + target_param.data * target_update_rate)
