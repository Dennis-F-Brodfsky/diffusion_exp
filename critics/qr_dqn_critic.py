from infrastructure.base_class import BaseCritic
import torch
from torch.nn import utils
from infrastructure import pytorch_util as ptu


class QRDQNCritic(BaseCritic):
    def __init__(self, params, **kwargs) -> None:
        super().__init__(**kwargs)
        self.double_q = params['double_q']
        self.grad_norm_clipping = params['max_norm_clipping']
        self.gamma = params['gamma']
        # self.use_entropy = params['use_entropy']
        self.target_update_rate = params['target_update_rate']
        self.q_net_spec = params['quantile_net_spec']
        self.q_net = params['quantile_func']()
        self.q_net_target = params['quantile_func']()
        self.loss = ptu.QuantileHuberLoss() # Quantile Huber loss
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.q_net_target.eval()
        # self.q_net_target.require_grad_(False)
        self.parameters = self.q_net.parameters()
        self.q_net_optimizer, self.q_net_scheduler = ptu.build_optim(self.q_net_spec, self.parameters)

    def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
        self.q_net.train()
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)  # with shape(batch_size, )
        next_ob_no = ptu.from_numpy(next_ob_no)
        reward_n = ptu.from_numpy(reward_n)
        terminal_n = ptu.from_numpy(terminal_n)
        
        # qa-values with shape (batch_size, n_quantiles, ac_dim)
        qa_t_values = self.q_net(ob_no) # qa refers quantile action values here
        qa_tp1_values = self.q_net_target(next_ob_no) # tp1 refers to t plus 1
        batch_size = ac_na.shape[0]
        n_quantile = qa_t_values.shape[1]
        # estimations
        q_t_values = torch.gather(qa_t_values, 2, ac_na[..., None, None].expand(batch_size, n_quantile ,1)).squeeze(2)
        if self.double_q:
            qa_t1_action = (self.q_net(next_ob_no)).mean(dim=1, keepdim=True).argmax(dim=2, keepdims=True)
            q_tp1 = qa_tp1_values.gather(2, qa_t1_action.expand(batch_size, n_quantile, 1)).squeeze(2)
        else:
            q_tp1, _ = qa_tp1_values.max(dim=2)
        # target / prediction
        target = reward_n.unsqueeze(-1) + self.gamma * q_tp1 * (1 - terminal_n).unsqueeze(-1)
        target = target.detach()
        loss = self.loss(q_t_values, target)
        self.q_net_optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.parameters, self.grad_norm_clipping)
        self.q_net_optimizer.step()
        if self.q_net_spec[2]:
            self.q_net_scheduler.step()
        return {'Delta Error': ptu.to_numpy(loss), 'Estimated Q': q_t_values.mean().item()}

    def update_target_network(self):
        self.soft_update(self.q_net, self.q_net_target, self.target_update_rate)

    @torch.no_grad()
    def qa_values(self, obs, **kwargs):
        self.q_net.eval()
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs).mean(dim=1)
        return ptu.to_numpy(qa_values)

    @torch.no_grad()
    def estimate_values(self, obs, policy, **kwargs):
        self.q_net.eval()
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs).mean(dim=1)
        return qa_values

    @staticmethod
    def soft_update(net, target_net, target_update_rate):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data * target_update_rate + target_param.data * (1-target_update_rate))
