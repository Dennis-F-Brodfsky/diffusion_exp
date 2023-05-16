import os
import numpy as np
from critics.dqn_critic import DQNCritic
from critics.qr_dqn_critic import QRDQNCritic
from policies.argmax_policy import ArgmaxPolicy
from infrastructure.base_class import BaseAgent
from policies.ppo_policy import MLPPolicyPPO
from infrastructure.utils import unnormalize, normalize, FlexibleReplayBuffer



class DiffusionPGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(DiffusionPGAgent, self).__init__()
        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']
        self.actor = MLPPolicyPPO(agent_params['ac_dim'], agent_params['mean_net'], agent_params['logits_na'],
                                    agent_params['max_norm_clipping'], agent_params['actor_optim_spec'], 
                                    agent_params['ppo_eps'], agent_params['discrete'])
        self.obs = []
        self.action = []
        self.reward = []
        self.terminal = []

    def train(self):
        all_logs = []
        for train_step in range(self.agent_params['num_agent_train_steps_per_iter']):
            ob_batch, ac_batch, rew_batch, terminal_batch = self.sample(self.agent_params['train_batch_size'])
            q_values = self.calculate_q_vals(rew_batch)
            adv_n = self.estimate_advantage(ob_batch, rew_batch, q_values, terminal_batch)
            # print(adv_n)
            train_log = self.actor.update(ob_batch, ac_batch, adv_n, q_values)
            all_logs.append(train_log)
        return all_logs[-1]

    def add_to_replay_buffer(self, paths, add_noised=False):
        self.obs, self.action, self.reward, self.terminal = [], [], [], []
        for path in paths:
            self.obs.append(path['observation'])
            self.action.append(path['action'])
            self.reward.append(path['reward'])
            self.terminal.append(path['terminal'])
        self.obs = np.concatenate(self.obs, axis=0, dtype=np.int8)
        self.action = np.concatenate(self.action, axis=0, dtype=np.int8)
        self.reward = self.reward
        self.terminal = np.concatenate(self.terminal, axis=0)
        # print(f'timesteps for inference: {np.mean(self.action)}')
        print(np.mean(self.action[:250]), np.mean(self.action[250:500]), np.mean(self.action[500:750]), np.mean(self.action[750:]))
        # print(self.reward)

    def sample(self, batch_size):
        return self.obs, self.action, self.reward, self.terminal

    def save(self, filepath):
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        self.actor.save(filepath + '/actor_info.pth')

    def calculate_q_vals(self, rewards_list):
        if not self.reward_to_go:
            q_values = np.concatenate([self._discounted_return(reward) for reward in rewards_list])
        else:
            q_values = np.concatenate([self._discounted_cumsum(reward) for reward in rewards_list])
        return q_values

    def estimate_advantage(self, obs, rews_list, q_values, terminals):
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            assert values_unnormalized.ndim == q_values.ndim
            values = unnormalize(normalize(values_unnormalized, np.mean(values_unnormalized), np.std(values_unnormalized)), np.mean(q_values), np.std(q_values))
            if self.gae_lambda is not None:
                values = np.append(values, [0])
                rews = np.concatenate(rews_list)
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)
                for i in reversed(range(batch_size)):
                    delta = rews[i] + (1-terminals[i])*self.gamma*values[i+1] - values[i]
                    advantages[i] = delta + (1-terminals[i])*self.gae_lambda*self.gamma*advantages[i+1]
                advantages = advantages[:-1]
            else:
                advantages = q_values - values
        else:
            advantages = q_values.copy()
        if self.standardize_advantages:
            advantages = normalize(advantages, np.mean(advantages), np.std(advantages))
        return advantages

    def _discounted_return(self, rewards):
        list_of_discounted_returns = np.full_like(rewards, fill_value=np.dot(rewards, self.gamma**np.arange(len(rewards))).item())
        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        list_of_discounted_cumsums = np.dot([[self.gamma**(col-row) if col >= row else 0 for col in range(len(rewards))] for row in range(len(rewards))], rewards)
        return list_of_discounted_cumsums


class DiffusionQAgent(BaseAgent):
    def __init__(self, env, agent_params):
        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.last_obs = self.env.reset()
        self.learning_start = agent_params['learning_start']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']
        self.exploration = agent_params['exploration_schedule']
        self.loc = agent_params['loc']
        self.scale = agent_params['scale']
        self.prefer_to_skip = agent_params['skip_prob']
        self.critic = DQNCritic(agent_params)
        self.actor = ArgmaxPolicy(self.critic)
        self.replay_buffer = FlexibleReplayBuffer(agent_params['buffer_size'], agent_params['horizon'])
        self.t = 0
        self.replay_buffer_idx = None
        self.latest_nfe = None
        self.latest_dis_score = None
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths, add_noised=False):
        pass

    def step_env(self):
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        eps = self.exploration.value(self.t)
        perform_random_action = eps > np.random.random() or self.t <= self.learning_start
        if perform_random_action:
            # action = self.env.action_space.sample()
            if np.random.random() < self.prefer_to_skip:
                action = 0
            else:
                action = 1
        else:
            ob = self.replay_buffer.encode_recent_observation()
            if not hasattr(ob, '__len__'):
                ob_batch, = np.array([ob])
            else:
                ob_batch = ob[None]
            action = self.actor.get_action(ob_batch)[0]
        obs, reward, done, _ = self.env.step(action)
        self.last_obs = obs.copy()
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)
        if done:
            self.latest_dis_score = reward / self.scale - self.loc
            self.latest_infrence_step = obs == 1
            print(self.latest_infrence_step.nonzero()[0][::-1])
            self.last_obs = self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(batch_size):
            return self.replay_buffer.sample_random_data(batch_size)
        else:
            return [], [], [], [], []

    def train(self):
        logs = {}
        for train_step in range(self.agent_params['num_agent_train_steps_per_iter']):
            train_log = {}
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.sample(self.agent_params['train_batch_size'])
            if (self.t > self.learning_start and self.t % self.learning_freq == 0
                    and self.replay_buffer.can_sample(self.batch_size)):
                train_log = self.critic.update(ob_batch, ac_batch, next_ob_batch, re_batch, terminal_batch)
                if self.num_param_updates % self.target_update_freq == 0:
                    self.critic.update_target_network()
                self.num_param_updates += 1
            self.t += 1
            logs.update(train_log)
        if self.latest_dis_score is not None and self.latest_infrence_step is not None:
            logs.update({'DIS_score': self.latest_dis_score, 
                         "NFE": np.sum(self.latest_infrence_step),
                         "NFE_0.25": np.sum(self.latest_infrence_step[:250]),
                         "NFE_0.5": np.sum(self.latest_infrence_step[250:500]),
                         "NFE_0.75": np.sum(self.latest_infrence_step[500:750]),
                         "NFE_1.0": np.sum(self.latest_infrence_step[750:])})

        return logs

    def save(self):
        pass


class DiffusionQRDQNAgent(DiffusionQAgent):
    def __init__(self, env, agent_params):
        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.last_obs = self.env.reset()
        self.learning_start = agent_params['learning_start']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']
        self.exploration = agent_params['exploration_schedule']
        self.loc = agent_params['loc']
        self.scale = agent_params['scale']
        self.prefer_to_skip = agent_params['skip_prob']
        self.critic = QRDQNCritic(agent_params)
        self.actor = ArgmaxPolicy(self.critic)
        self.replay_buffer = FlexibleReplayBuffer(agent_params['buffer_size'], agent_params['horizon'])
        self.t = 0
        self.replay_buffer_idx = None
        self.latest_nfe = None
        self.latest_dis_score = None
        self.num_param_updates = 0
