import gym
from gym.spaces import Discrete, MultiBinary
import torch
import numpy as np
import infrastructure.pytorch_util as ptu
from diffusers import DDPMPipeline


class DiffusionEnv(gym.Env):
    def __init__(self, params) -> None:
        super().__init__()
        self.DIS = params['dis'].to(ptu.device)
        self.action_space = Discrete(2)
        self.observation_space = MultiBinary(1000)
        self.diffuser = DDPMPipeline.from_pretrained(params['diffuser_dir']).to(ptu.device)
        self.logdir = params['logdir']
        self.scheduler = params['diffuser_scheduler'].from_config(self.diffuser.scheduler.config)
        self.unet = self.diffuser.unet
        self.num_inference_steps = params['num_inference_steps']
        self.cur_t = self.num_inference_steps - 1
        self.state = np.zeros(self.num_inference_steps, dtype=np.int8)
        self.img_size_per_batch = (params['inference_batch_size'],)+params['image_size']
        self.penalty = params['penalty']
        self.loc = params['loc']
        self.scale = params['scale']
        self.n_itr = params['time_steps']

    @torch.no_grad()
    def step(self, action):
        done = self.cur_t == 0
        if action == 0:
            if done:
                return self._process_after_done()
            self.state[self.cur_t] = -1
            self.cur_t -= 1
            return self.state[:], torch.randn(1).item()*0.01, done, {}
        else:
            self.state[self.cur_t] = 1
            self.cur_t -= 1
            if done:
                return self._process_after_done()
            return self.state[:], -self.penalty, done, {}

    def reset(self):
        torch.cuda.empty_cache()
        self.cur_t = self.num_inference_steps - 1
        self.state = np.zeros(self.num_inference_steps, dtype=np.int8)
        return self.state[:]

    def close(self) -> None:
        return super().close()

    @torch.no_grad()
    def _process_after_done(self):
        # t1 = time.time()
        self.scheduler.set_timesteps(ptu.device, self.state)
        image = torch.randn(self.img_size_per_batch)
        for t in self.scheduler.timesteps:
            model_output = self.unet(image.to(ptu.device), t).sample
            image = self.scheduler.step(model_output, t, image.to(ptu.device)).prev_sample
        # print('diffusion inference cost:', time.time()- t1)
        return self.state[:], (self.DIS(image.to(ptu.device), 1)['adv_output'].mean().item()+self.loc)*self.scale, True, {}


class DiffusionEnv_v2(gym.Env):
    def __init__(self, params) -> None:
        super().__init__()
        self.DIS = params['dis'].to(ptu.device)
        self.action_space = Discrete(1000)
        self.observation_space = MultiBinary(1000)
        self.diffuser = DDPMPipeline.from_pretrained(params['diffuser_dir']).to(ptu.device)
        self.logdir = params['logdir']
        self.scheduler = params['diffuser_scheduler'].from_config(self.diffuser.scheduler.config)
        self.unet = self.diffuser.unet
        self.num_inference_steps = params['num_inference_steps']
        self.cur_t = 0
        self.state = np.zeros(self.num_inference_steps, dtype=np.int8)
        self.img_size_per_batch = (params['inference_batch_size'],)+params['image_size']
        self.penalty = params['penalty']
        self.loc = params['loc']
        self.scale = params['scale']
        self.n_itr = params['time_steps']
    
    @torch.no_grad()
    def step(self, action):
        self.state[self.cur_t:self.cur_t+action+1] = -1
        self.cur_t += (action + 1)  # map 0-999 to 1-1000
        done = self.cur_t >= self.num_inference_steps
        if done:
            return self._process_after_done()
        self.state[self.cur_t] = 1
        return self.state[:], -self.penalty, done, {}
    
    @torch.no_grad()
    def _process_after_done(self):
        # t1 = time.time()
        self.scheduler.set_timesteps(ptu.device, self.state)
        image = torch.randn(self.img_size_per_batch)
        for t in self.scheduler.timesteps:
            model_output = self.unet(image.to(ptu.device), t).sample
            image = self.scheduler.step(model_output, t, image.to(ptu.device)).prev_sample
        # print('diffusion inference cost:', time.time()- t1)
        return self.state[:], (self.DIS(image.to(ptu.device), 1)['adv_output'].mean().item()+self.loc)*self.scale, True, {}
    
    def reset(self):
        torch.cuda.empty_cache()
        self.cur_t = -1 # initial-state
        self.state = np.zeros(self.num_inference_steps, dtype=np.int8)
        return self.state[:]
