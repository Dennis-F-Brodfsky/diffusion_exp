import pickle
import time
from collections import OrderedDict
from typing import Dict
import os
import gym
import numpy as np
import torch
import agents
import envs
from infrastructure import pytorch_util as ptu
from infrastructure import utils
from infrastructure.logger import Logger
import random


_str_to_env = {name + '-v0': getattr(envs, name) for name in envs.__all__}
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RLTrainer:
    def __init__(self, params: Dict):
        # get param and create Logger
        self.params = params
        self.logger = Logger(self.params['logdir'])
        self.log_condition = False
        self.logvideo = False
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params['ep_len']

        # random seed and init gpu
        seed = self.params['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(not self.params['no_gpu'], self.params['which_gpu'])
        # create env and setup some env information
        self.env = self._create_env(params)
        self.eval_env = self._create_eval_env(params)
        params['action_space'] = self.env.action_space
        if isinstance(self.env, gym.Env):
            self.params['ac_dim'] = self._get_object_dim(self.env.action_space)
            self.params['obs_dim'] = self._get_object_dim(self.env.observation_space)
        else:
            self.params['ac_dim'] = self.env.ac_dim
            self.params['obs_dim'] = self.env.obs_dim
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        self.params['discrete'] = isinstance(self.env.action_space, gym.spaces.Discrete)
        # other setting
        self.start_time = self.total_envsteps = 0
        self.agent_class = params['agent_class']
        self.agent = self.agent_class(self.env, self.params)

        if 'model' in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        elif 'env_wrappers' in self.params:
            self.fps = 30  # This is not actually used when using the Monitor wrapper
        elif hasattr(self.env, 'env') and 'video.frames_per_second' in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata['video.frames_per_second']
        else:
            self.fps = 10

    @staticmethod
    def do_relabel_with_expert(expert_policy, paths):
        for i in range(len(paths)):
            paths[i]["action"] = expert_policy.get_action(paths[i]['observation'])
        return paths

    def collect_training_trajectory(self, itr: int, initial_expertdata: str, collect_policy, num_transitions_to_sample):
        if itr == 0 and initial_expertdata:
            with open(initial_expertdata, 'rb') as f:
                paths = pickle.load(f)
                return paths, 0, None
        paths, envsteps_this_batch = utils.sample_trajectories(self.env, collect_policy, num_transitions_to_sample,
                                                               self.params['ep_len'])
        train_video_paths = None
        if self.logvideo:
            # print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
        return paths, envsteps_this_batch, train_video_paths

    def _create_env(self, params) -> gym.Env:
        if self.params['env_name'] in _str_to_env:
            env = _str_to_env[self.params['env_name']](**params['env_config'])
        env.seed(params['seed'])
        return env

    def _create_eval_env(self, params) -> gym.Env:
        return self._create_env(params)  # if env_wrappers != None, rewrite thie method.

    @staticmethod
    def _get_object_dim(obj):
        if isinstance(obj, gym.spaces.Discrete) or isinstance(obj, gym.spaces.MultiBinary):
            return obj.n
        elif isinstance(obj, gym.spaces.Box):
            return obj.shape[0]  # if problem raised, rewrite this.
        else:
            raise "may implement in future version"

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs):
        self._video_log(itr, self.eval_env, eval_policy, train_video_paths)
        if self.log_condition:  # logging condition
            print(f"\n\n---------------iter {itr} -------------------:")
            logs = OrderedDict()
            logs = self._train_log(logs, paths)
            logs = self._eval_log(logs, self.eval_env, eval_policy)
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(training_logs[-1])
            for key, value in logs.items():
                print(f"{key}: {value}")
                self.logger.log_scalar(value, key, itr)
            self.logger.flush()

    def _video_log(self, itr, eval_env, eval_policy, train_video_paths):
        if self.logvideo and train_video_paths is not None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(eval_env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)
            # save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='eval_rollouts')

    def _eval_log(self, logs, eval_env, eval_policy):
        eval_paths, eval_env_step = utils.sample_trajectories(eval_env, eval_policy, self.params['eval_batch_size'],
                                                              self.params['ep_len'])
        eval_return = [eval_path['reward'].sum() for eval_path in eval_paths]
        eval_dis_score = [eval_path['reward'][-1] for eval_path in eval_paths]
        # eval_eplen = [len(eval_path['reward']) for eval_path in eval_paths]
        eval_num_inference_steps = [eval_path['action'].sum() for eval_path in eval_paths]
        logs["NFE"] = np.mean(eval_num_inference_steps)
        logs["Eval_AverageReturn"] = np.mean(eval_return)
        logs["Eval_StdReturn"] = np.std(eval_return)
        logs["Eval_MaxReturn"] = np.max(eval_return)
        logs["Eval_MinReturn"] = np.min(eval_return)
        logs["DIS_SCORE"] = np.mean(eval_dis_score)
        # logs["Eval_AverageEpLen"] = np.mean(eval_eplen)
        return logs

    def _train_log(self, logs, paths):
        train_return = [path['reward'].sum() for path in paths]
        # train_eplen = [len(path['reward']) for path in paths]
        logs["Train_AverageReturn"] = np.mean(train_return)
        logs["Train_StdReturn"] = np.std(train_return)
        logs["Train_MaxReturn"] = np.max(train_return)
        logs["Train_MinReturn"] = np.min(train_return)
        # logs["Train_AverageEpLen"] = np.mean(train_eplen)
        logs["Train_EnvstepsSoFar"] = self.total_envsteps
        return logs

    def run_training_loop(self, n_iter: int, collect_policy, eval_policy, initial_expert_data=None,
                          relabel_with_expert=False, start_relabel_with_expert=1, expert_policy=None):
        self.start_time = time.time()
        self.total_envsteps = 0
        for itr in range(n_iter):
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.logvideo = True
            else:
                self.logvideo = False
            if self.params['scalar_log_freq'] == -1:
                self.log_condition = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.log_condition = True
            else:
                self.log_condition = False
            if itr == 0:
                use_batch_size = self.params['batch_size_initial']
            else:
                use_batch_size = self.params['batch_size']
            paths, step, train_video_paths = self.collect_training_trajectory(itr, initial_expert_data,
                                                                              collect_policy, use_batch_size)
            self.total_envsteps += step
            if relabel_with_expert and itr >= start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)
            self.agent.add_to_replay_buffer(paths, self.params['add_ob_noise'])
            training_logs = self.agent.train()
            if self.log_condition or self.logvideo:  # log condition
                self.perform_logging(itr, paths, eval_policy, train_video_paths, [training_logs])
                if self.params['save_params']:
                    self.agent.save(f'{self.params["logdir"]}/agent_{itr}')
        # TODO eval 10000 picture to calc FID-score



class DiffustionRLTrainer(RLTrainer):
    def __init__(self, params: Dict):
        params['agent_class'] = agents.DiffusionPGAgent
        super().__init__(params)

    def _create_env(self, params) -> gym.Env:
        env = envs.DiffusionEnv(params)
        env.seed(params['seed'])
        return env

    def _create_eval_env(self, params) -> gym.Env:
        env = envs.DiffusionEnv(params, True)
        env.seed(params['seed'])
        return env


class DiffusionQTrainer(RLTrainer):
    def __init__(self, params: Dict):
        params['agent_class'] = agents.DiffusionQAgent
        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')
        super().__init__(params)
        self.eval_env = self._create_eval_env()

    def _create_env(self, params) -> gym.Env:
        env = envs.DiffusionEnv(params)
        if 'env_wrappers' in self.params:
            env = envs.Monitor(
                env,
                os.path.join(self.params['logdir'], "gym"),
                force=True,
                video_callable=(None if self.params['video_log_freq'] > 0 else False),
            )
            env = params['env_wrappers'](env)
        env.seed(params['seed'])
        return env

    def _create_eval_env(self, params) -> gym.Env:
        env = envs.DiffusionEnv(params, True)
        env.seed(params['seed'])
        return env
    
    def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs):
        last_log = training_logs[-1]
        episode_rewards = self.__get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            self.mean_episode_reward = np.mean(episode_rewards[-100:]).item()
        if len(episode_rewards) > 100:
            self.best_mean_episode_reward = max(self.best_mean_episode_reward, self.mean_episode_reward)
        logs = OrderedDict()
        logs["Train_EnvstepsSoFar"] = self.agent.t
        print("Timestep %d" % (self.agent.t,))
        if self.mean_episode_reward > -5000:
            logs["Train_AverageReturn"] = np.mean(self.mean_episode_reward)
        print("mean reward (100 episodes) %f" % self.mean_episode_reward)
        if self.best_mean_episode_reward > -5000:
            logs["Train_BestReturn"] = np.mean(self.best_mean_episode_reward)
        print("best mean reward %f" % self.best_mean_episode_reward)
        if self.start_time is not None:
            time_since_start = (time.time() - self.start_time)
            print("running time %f" % time_since_start)
            logs["TimeSinceStart"] = time_since_start
        logs.update(last_log)
        for key, value in logs.items():
            print('{} : {}'.format(key, value))
            self.logger.log_scalar(value, key, self.agent.t)
        print('Done logging...\n\n')
        self.logger.flush()

    @staticmethod
    def __get_wrapper_by_name(env, name):
        currentenv = env
        while True:
            if name in currentenv.__class__.__name__:
                return currentenv
            elif isinstance(env, gym.Wrapper):
                currentenv = currentenv.env
            else:
                raise ValueError("Couldn't find wrapper named %s" % name)
            
    def collect_training_trajectory(self, itr: int, initial_expertdata: str, collect_policy, num_transitions_to_sample):
        self.agent.step_env()
        return None, 1, None
