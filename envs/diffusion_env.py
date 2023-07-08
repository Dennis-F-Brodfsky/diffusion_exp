import gym
from gym.spaces import Discrete, MultiBinary
import torch
import numpy as np
import infrastructure.pytorch_util as ptu
from infrastructure.utils import numpy_to_pil
from diffusers import DDPMPipeline
# from diffusers.schedulers.scheduling_utils import SchedulerOutput

class DiffusionEnv(gym.Env):
    def __init__(self, params, is_eval=False) -> None:
        super().__init__()
        self.DIS = params['dis'].to(ptu.device)
        self.action_space = Discrete(2)
        self.observation_space = MultiBinary(1000)
        self.diffuser = DDPMPipeline.from_pretrained(params['diffuser_dir']).to(ptu.device)
        self.logdir = params['logdir']
        self._is_eval = is_eval
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
        self.idx = 0
        self.img_idx = 0

    @torch.no_grad()
    def step(self, action):
        done = self.cur_t == 0
        if action == 0:
            if done:
                return self._process_after_done()
            self.state[self.cur_t] = -1
            self.cur_t -= 1
            return self.state[:], torch.randn(1).item()*0.01, done, {}
            # return self.cur_t, 0, done, {}
        else:
            self.state[self.cur_t] = 1
            self.cur_t -= 1
            if done:
                return self._process_after_done()
            return self.state[:], -self.penalty, done, {}
            # return self.cur_t, -self.penalty, done, {}

    def reset(self):
        torch.cuda.empty_cache()
        self.cur_t = self.num_inference_steps - 1
        self.state = np.zeros(self.num_inference_steps, dtype=np.int8)
        return self.state[:]
        # return self.cur_t

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
        # 输出PIL图片，转化
        # print('diffusion inference cost:', time.time()- t1)
        self.idx += 1
        if self._is_eval: # or self.idx >= self.n_itr // 1000 - 50:
            result_img = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
            result_img = numpy_to_pil(result_img)
            for img in result_img:
                self.img_idx += 1
                img.save(self.logdir + f'/img/{self.img_idx}.jpg')
        return self.state[:], (self.DIS(image.to(ptu.device), 1)['adv_output'].mean().item()+self.loc)*self.scale, True, {}
