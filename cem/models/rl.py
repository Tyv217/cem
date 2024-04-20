import math
from typing import Dict, Tuple

import gymnasium as gym
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from torch import Tensor
from torch.distributions import Categorical
from torchmetrics import MeanMetric


class PPOLightningAgent(pl.LightningModule):
    def __init__(
        self,
        envs: gym.vector.SyncVectorEnv,
        act_fun: str = "relu",
        ortho_init: bool = False,
        vf_coef: float = 1.0,
        ent_coef: float = 0.0,
        clip_coef: float = 0.2,
        clip_vloss: bool = False,
        normalize_advantages: bool = False,
        **torchmetrics_kwargs,
    ):
        super().__init__()
        if act_fun.lower() == "relu":
            act_fun = torch.nn.ReLU()
        elif act_fun.lower() == "tanh":
            act_fun = torch.nn.Tanh()
        else:
            raise ValueError("Unrecognized activation function: `act_fun` must be either `relu` or `tanh`")
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.normalize_advantages = normalize_advantages
        self.critic = torch.nn.Sequential(
            layer_init(
                torch.nn.Linear(math.prod(self.get_shape(envs.single_observation_space)), 64),
                ortho_init=ortho_init,
            ),
            act_fun,
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, 1), std=1.0, ortho_init=ortho_init),
        )
        self.actor = torch.nn.Sequential(
            layer_init(
                torch.nn.Linear(math.prod(self.get_shape(envs.single_observation_space)), 64),
                ortho_init=ortho_init,
            ),
            act_fun,
            layer_init(torch.nn.Linear(64, 64), ortho_init=ortho_init),
            act_fun,
            layer_init(torch.nn.Linear(64, envs.single_action_space_shape), std=0.01, ortho_init=ortho_init),
        )
        self.avg_pg_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_value_loss = MeanMetric(**torchmetrics_kwargs)
        self.avg_ent_loss = MeanMetric(**torchmetrics_kwargs)

    def get_shape(self, dict):
        # total_shape = 0
        # for key, value in dict.items():
        #     shape = value.shape
        #     if len(shape) > 1:
        #         raise ValueError("Get shape unable to flatten 2d shape")
        #     elif len(shape) == 0:
        #         total_shape += 1
        #     else:
        #         total_shape += shape[0]
        
        # return (total_shape, )
        return dict.shape

    def dict_to_tensor(self, obs, device='cpu'):
        tensors = []
        for _, value in obs.items():
            if isinstance(value, np.ndarray):
                tensor = torch.tensor(value, dtype = torch.float32, device = device)
            elif isinstance(value, (int, float)):  # Scalar values for discrete spaces
                tensor = torch.tensor([value], dtype = torch.float32, device = device)
            tensors.append(tensor)
        if len(tensors) > 1:
            final_tensor = torch.cat(tensors, dim=-1)
        else:
            final_tensor = tensors[0]
        return final_tensor

    def get_action(self, x: dict, action: Tensor = None, info = None, train = False) -> Tuple[Tensor, Tensor, Tensor]:
        logits = self.actor(x)
        if not train:
            action_mask = info["action_mask"]
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask).to(x.device)
            logits = torch.where(action_mask.bool(), logits, -1000)
        distribution = Categorical(logits=logits)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy()

    def get_greedy_action(self, x: Tensor) -> Tensor:
        logits = self.actor(x)
        probs = F.softmax(logits, dim=-1)
        return torch.argmax(probs, dim=-1)

    def get_value(self, x: Tensor) -> Tensor:
        return self.critic(x)

    def get_action_and_value(self, x: Tensor, action: Tensor = None, info = None, train = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        action, log_prob, entropy = self.get_action(x, action, info, train)
        value = self.get_value(x)
        return action, log_prob, entropy, value

    def forward(self, x: Tensor, action: Tensor = None, train = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.get_action_and_value(x, action, train = train)

    @torch.no_grad()
    def estimate_returns_and_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        next_obs: Tensor,
        next_done: Tensor,
        num_steps: int,
        batch_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[Tensor, Tensor]:
        next_value = self.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            t_1 = t * batch_size
            t_2 = (t + 1) * batch_size
            t_3 = (t + 2) * batch_size
            if t == num_steps - 1:
                nextnonterminal = torch.logical_not(next_done)
                nextvalues = next_value
            else:
                nextnonterminal = torch.logical_not(dones[t_2:t_3])
                nextvalues = values[t_2:t_3]
            delta = rewards[t_1:t_2] + gamma * nextvalues * nextnonterminal - values[t_1:t_2]
            advantages[t_1:t_2] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        # [batch_size * num_steps]

        return returns, advantages

    def training_step(self, batch: Dict[str, Tensor]):
        # Get actions and values given the current observations
        _, newlogprob, entropy, newvalue = self(batch["obs"], batch["actions"].long(), train = True)
        logratio = newlogprob - batch["logprobs"]
        ratio = logratio.exp()

        # Policy loss
        advantages = batch["advantages"]
        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        pg_loss = policy_loss(batch["advantages"], ratio, self.clip_coef)

        # Value loss
        v_loss = value_loss(
            newvalue,
            batch["values"],
            batch["returns"],
            self.clip_coef,
            self.clip_vloss,
            self.vf_coef,
        )

        # Entropy loss
        ent_loss = entropy_loss(entropy, self.ent_coef)

        # Update metrics
        self.avg_pg_loss(pg_loss)
        self.avg_value_loss(v_loss)
        self.avg_ent_loss(ent_loss)

        # Overall loss
        return pg_loss + ent_loss + v_loss

    def on_train_epoch_end(self, global_step: int) -> None:
        # Log metrics and reset their internal state
        self.logger.log_metrics(
            {
                "Loss/policy_loss": self.avg_pg_loss.compute(),
                "Loss/value_loss": self.avg_value_loss.compute(),
                "Loss/entropy_loss": self.avg_ent_loss.compute(),
            },
            global_step,
        )
        self.reset_metrics()

    def reset_metrics(self):
        self.avg_pg_loss.reset()
        self.avg_value_loss.reset()
        self.avg_ent_loss.reset()

    def configure_optimizers(self, lr: float):
        return torch.optim.Adam(self.parameters(), lr=lr, eps=1e-4)
    


def policy_loss(advantages: torch.Tensor, ratio: torch.Tensor, clip_coef: float) -> torch.Tensor:
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
    return torch.max(pg_loss1, pg_loss2).mean()


def value_loss(
    new_values: Tensor,
    old_values: Tensor,
    returns: Tensor,
    clip_coef: float,
    clip_vloss: bool,
    vf_coef: float,
) -> Tensor:
    new_values = new_values.view(-1)
    if not clip_vloss:
        values_pred = new_values
    else:
        values_pred = old_values + torch.clamp(new_values - old_values, -clip_coef, clip_coef)
    return vf_coef * F.mse_loss(values_pred, returns)


def entropy_loss(entropy: Tensor, ent_coef: float) -> Tensor:
    return -entropy.mean() * ent_coef

def layer_init(
    layer: torch.nn.Module,
    std: float = math.sqrt(2),
    bias_const: float = 0.0,
    ortho_init: bool = True,
):
    if ortho_init:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer