import dataclasses
from infrastructure.base_class import Schedule
from typing import Optional, Sequence, Tuple, Union, Callable
from torch.nn import Module
from infrastructure.utils import OptimizerSpec


@dataclasses.dataclass
class BasicConfig:
    env_name: str
    time_steps: int
    batch_size: int = 4000
    ep_len: int = 1000
    eval_batch_size: int = 1000
    buffer_size: int = int(1e6)
    seed: int = 1
    no_gpu: bool = False
    which_gpu: int = 0
    exp_name: str = "todo"
    save_params: bool = False
    scalar_log_freq: int = -1
    video_log_freq: int = -1
    num_agent_train_steps_per_iter: int = 1
    logdir: str = None
    batch_size_initial: int = batch_size
    add_ob_noise: bool = False
    env_wrappers: Optional[Callable] = None
    dis: object = None
    penalty: float = 0.05
    num_inference_steps: int = 1000
    loc: float = 0
    scale: float = 1
    image_size: Tuple = (3, 32, 32)
    inference_batch_size: int = 4
    diffuser_scheduler: Union[Module, None] = None
    diffuser_dir: str = None
    gan_dir: str = None


@dataclasses.dataclass
class DiffusionConfig(BasicConfig):
    max_norm_clipping: float = 10
    gamma: float = 0.9
    mean_net: Union[Module, None] = None
    logits_na: Union[Module, None] = None
    baseline_network: Union[Module, None] = None
    actor_optim_spec: Union[OptimizerSpec, None] = None
    baseline_optim_spec: Union[OptimizerSpec, None] = None
    standardize_advantages: bool = False
    reward_to_go: bool = False
    nn_baseline: bool = False
    gae_lambda: float = None
    action_noise_std: float = 0
    horizon: int = 1
    ppo_eps: float = 0

    def __post_init__(self):
        self.train_batch_size = self.batch_size
        assert bool((self.mean_net or self.logits_na) and self.actor_optim_spec)
        if self.nn_baseline:
            assert bool(self.baseline_network and self.baseline_optim_spec)


@dataclasses.dataclass
class DiffusionQConfig(BasicConfig):
    skip_prob: float = 0.75
    max_norm_clipping: float = 10
    learning_freq: int = 1
    learning_start: int = int(5e3)
    target_update_freq: int = int(1e3)
    target_update_rate: float = 0.95
    q_func: Union[Callable, None] = None
    q2_func: Optional[Callable] = None
    clipped_q: bool = False
    double_q: bool = False
    exploration_schedule: Schedule = None
    q_net_spec: OptimizerSpec = None
    env_wrappers: Callable = None
    gamma: float = 0.99
    horizon: int = 1

    def __post_init__(self):
        self.train_batch_size = self.batch_size
        if self.clipped_q:
            assert bool(self.q2_func)


@dataclasses.dataclass
class DiffusionQRDQNConfig(BasicConfig):
    skip_prob: float = 0.75
    max_norm_clipping: float = 10
    learning_freq: int = 1
    learning_start: int = int(5e3)
    target_update_freq: int = int(1e3)
    target_update_rate: float = 0.95
    quantile_func: Union[Callable, None] = None
    double_q: bool = False
    exploration_schedule: Schedule = None
    quantile_net_spec: OptimizerSpec = None
    env_wrappers: Callable = None
    gamma: float = 0.99
    horizon: int = 1

    def __post_init__(self):
        self.train_batch_size = self.batch_size
