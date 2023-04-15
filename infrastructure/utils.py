import os
import copy
import numpy as np
from PIL import Image
from infrastructure.base_class import Schedule
import time
import random
from collections import namedtuple

OptimizerSpec = namedtuple('OptimizerSpec', ["constructor", "optim_kwargs", "learning_rate_schedule"])


def set_config_logdir(config):
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')
    config.logdir = config.exp_name + '_' + config.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    config.logdir = os.path.join(data_path, config.logdir)
    if not os.path.exists(config.logdir):
        os.makedirs(config.logdir)
    if not os.path.exists(config.logdir + '/img'):
        os.makedirs(config.logdir + '/img')


def sample_discrete(p):
    c = np.cumsum(p, axis=1)
    u = np.random.rand(len(c), 1)
    return np.argmax(u < c, axis=1)


def softmax(m):
    col_max = np.max(m, axis=1, keepdims=True)
    exp_partial = np.exp(m - col_max)
    return exp_partial / np.sum(exp_partial, axis=1, keepdims=True)


def serialize(*args):
    serial_arg = tuple()
    for arg in args:
        if not hasattr(arg, '__len__'):
            serial_arg += (np.array([arg]),)
        else:
            serial_arg += (np.array(arg),)
    return serial_arg


def registry_custom_env():
    from gym.envs.registration import register
    register(id='KArmBandit-v0',
             entry_point='TorchRL.envs.MultiArmBandit.KArmBandit')
    register(id='KArmNonStationaryBandit-v0',
             entry_point='TorchRL.envs.MultiArmBandit.KArmNonStationaryBandit')
    register(id='MarkovEnv-v0',
             entry_point='TorchRL.envs.MarkovEnv.MarkovProcess')


def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array',)):
    ob = env.reset()
    obs, acs, rewards, terminals, image_obs = [], [], [], [], []
    steps = 0
    while True:
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)
        obs.append(ob)
        # deal with single ob, increase the dim
        if not hasattr(ob, '__len__'):
            ob_batch, = serialize(ob)
        else:
            ob_batch = ob[None]
        ac = policy.get_action(ob_batch)
        ac = ac[0]
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        steps += 1
        rewards.append(rew)
        rollout_done = int(done or steps == max_path_length)
        terminals.append(rollout_done)
        if rollout_done:
            break
    return Path(obs, image_obs, acs, rewards, terminals)


def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False,
                        render_mode=('rgb_array',)):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        timesteps_this_batch += get_pathlength(path)
        paths.append(path)
    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array',)):
    paths = [sample_trajectory(env, policy, max_path_length, render, render_mode) for _ in range(ntraj)]
    return paths


def Path(obs, image_obs, acs, rewards, terminals):
    if image_obs:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation": np.array(obs),
            "image_obs": np.array(image_obs, dtype=np.uint8),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, terminals, concatenated_rewards, unconcatenated_rewards


def get_pathlength(path):
    return len(path["reward"])


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def add_noise(data_inp, noiseToSignal=0.01):
    data = copy.deepcopy(data_inp)
    mean_data = np.mean(data, axis=0)
    mean_data[mean_data == 0] = 0.000001
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(data[:, j] + np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],)))
    return data


class ConstantSchedule(Schedule):
    def __init__(self, value):
        self._v = value

    def value(self, t):
        return self._v


class PiecewiseSchedule(Schedule):
    def __init__(self, endpoints, outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return l + alpha * (r - l)
        assert self._outside_value is not None
        return self._outside_value


class LinearSchedule(Schedule):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


def sample_n_unique(sampling_f, n):
    """
    deprecated.... when n is large, the function suffer slow speed quite a lot!!!!!
    also it doesn't support prior sampling.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


str_to_schedule = {'Constant': ConstantSchedule,
                   'Linear': LinearSchedule,
                   'Piecewise': PiecewiseSchedule}


class FlexibleReplayBuffer(object):
    def __init__(self, size, horizon, encode_effect=False) -> None:
        self.size = size
        self.next_idx = 0
        self.horizon = horizon
        self.encode_effect = encode_effect
        self.num_in_buffer = 0
        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        return batch_size + 1 <= self.num_in_buffer

    def sample_random_data(self, batch_size):
        assert self.can_sample(batch_size)
        idxes = random.sample(range(self.num_in_buffer-1), batch_size)
        # idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def sample_recent_data(self, batch_size, concat_rew=True):
        assert self.can_sample(batch_size)
        if concat_rew:
            return self._encode_sample(
                idxes=[idx % self.size for idx in range(self.next_idx - batch_size - 1, self.next_idx - 1)])
        else:
            cur, i, unconcat_rew = (self.next_idx - 2) % self.size, 0, []
            while i < batch_size:
                if not self.done[cur]:
                    episode_rew = []
                else:
                    episode_rew = [self.reward[cur]]
                    i += 1
                    cur = (cur - 1) % self.size
                while i < self.num_in_buffer and not self.done[cur]:
                    episode_rew.append(self.reward[cur])
                    cur = (cur - 1) % self.size
                    i += 1
                unconcat_rew.append(episode_rew[::-1])
            data_range = range(self.next_idx - i - 1, self.next_idx - 1)
            return (np.stack([self._encode_observation(idx) for idx in data_range]),
                    self.action[data_range], unconcat_rew[::-1],
                    np.stack([self._encode_observation(idx + 1) for idx in data_range]),
                    self.done[data_range].astype(np.float32))

    def _encode_trajectory(self, idx):
        pass

    def _encode_observation(self, idx):
        # return with shape obs.shape when horizon == 1 and (horizon, obs.shape) when horizon > 1
        end_idx = idx + 1
        start_idx = end_idx - self.horizon
        if self.horizon == 1:
            return self.obs[(end_idx - 1)%self.size]
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for i in range(idx - 1, start_idx - 1, -1):
            if self.done[i % self.size]:
                start_idx = i + 1
                break
        missing_context = self.horizon - (end_idx - start_idx)
        if start_idx < 0 or missing_context > 0:
            # padding
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            # concatenate
            for i in range(start_idx, end_idx):
                frames.append(self.obs[i % self.size])
            return np.stack(frames, 0)
        else:
            return self.obs[start_idx:end_idx]

    def encode_recent_observation(self):
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_sample(self, idxes):
        if self.encode_effect:
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, attn_mask = self._encode_trajectory
            return obs_batch, act_batch, rew_batch, done_batch, attn_mask
        else:
            obs_batch = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
            next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
            act_batch = self.action[idxes]
            rew_batch = self.reward[idxes]
            done_batch = self.done[idxes].astype(np.float32)
            # done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
            return obs_batch, act_batch, rew_batch, next_obs_batch, done_batch

    def add_rollouts(self, paths, noised=False):
        for path in paths:
            if noised:
                observation = add_noise(path['observation'])
            else:
                observation = copy.deepcopy(path['observation'])
            for ob, ac, rew, done in zip(observation, path['action'], path['reward'], path['terminal']):
                idx = self.store_frame(ob)
                self.store_effect(idx, ac, rew, done)

    def store_frame(self, frame):
        if self.obs is None:
            if self.obs is None:
                if hasattr(frame, 'shape'):
                    self.obs = np.empty([self.size] + list(frame.shape), dtype=frame.dtype)
                else:
                    self.obs = np.empty([self.size], dtype=type(frame))
        self.obs[self.next_idx] = frame
        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
        return ret

    def store_effect(self, idx, action, reward, done):
        if self.action is None:
            if hasattr(action, 'shape'):
                self.action = np.empty([self.size] + list(action.shape), dtype=action.dtype)
            else:
                self.action = np.empty([self.size], dtype=type(action))
            self.reward = np.empty([self.size], dtype=np.float32)
            self.done = np.empty([self.size], dtype=np.bool)
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done
