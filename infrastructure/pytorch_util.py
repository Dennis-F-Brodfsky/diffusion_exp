from typing import Union
import torch
from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
        init_method=None
) -> nn.Module:
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size
    last_layer = nn.Linear(in_size, output_size)
    if init_method is not None:
        last_layer.apply(init_method)
    layers.append(last_layer)
    layers.append(output_activation)
    return nn.Sequential(*layers)


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


class Scalar(nn.Module):
    def __init__(self, val=0, requires_grad=True):
        super().__init__()
        self.value = nn.Parameter(data=torch.Tensor([val]), requires_grad=requires_grad)

    def forward(self):
        return self.value


def build_optim(optim_spec, param):
    optimizer = optim_spec[0](param, **optim_spec[1])
    lr_schedule = None
    if optim_spec[2]:
        lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, optim_spec[2])
    return optimizer, lr_schedule


def evaluate_rollout(img_0, noise_0, timestep, scheduler, unet):
    pass 

def transfer_01_timestep(action):
    return len(action) - 1 - torch.where(action)[0]


class QuantileHuberLoss(nn.Module):
    def __init__(self, k: float=1.0, is_sum_over: bool=True) -> None:
        super().__init__()
        self.k = k
        self.is_sum_over = is_sum_over
        self.cum_prob = None

    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        # expected shape: input = (batch_size, n_quantile); target = (batch_size, n_quantile)
        n = input_tensor.shape[-1]
        if self.cum_prob is None:
            self.cum_prob = (torch.arange(n, device=input_tensor.device, dtype=torch.float) + 0.5) / n
        abs_delta = torch.abs(target_tensor.unsqueeze(-2) - input_tensor.unsqueeze(-1))
        huber_loss = torch.where(abs_delta > self.k, abs_delta - 0.5, abs_delta**2*0.5)
        loss = torch.abs(self.cum_prob - (abs_delta.detach() < 0).float())*huber_loss
        if self.is_sum_over:
            return loss.sum(-2).mean()
        else:
            return loss.mean()
