
from typing import Any, List, Union
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from citylearn.agents.rlc import RLC
from citylearn.citylearn import CityLearnEnv



class PolicyGradient(RLC):
    """
    "Policy Gradient" agent pour CityLearn
    Uniquement compatible avec un agent centralisé (central_agent=True)
    """
    def __init__(self, env: CityLearnEnv, **kwargs: Any) -> None:
        super().__init__(env, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.action_low, self.action_high = self._compute_action_bounds(env)

        self._action_sizes = [space.shape[0] for space in self.action_space] \
            if isinstance(self.action_space, (list, tuple)) else [self.action_space.shape[0]] # test

        self._traj_rewards = []     # List[float] (un scalaire par step)
        self._traj_log_probs = []   # List[Tensor-scalar] (un par step)
        self.policy_network = self.build_policy_network(state_dim=sum(obs.shape[0] for obs in self.observation_space), action_dim=sum(space.shape[0] for space in self.action_space), hidden_dims=[128, 128]).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=3e-4)
        self.gamma = 0.99

    def build_policy_network(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> nn.Module:
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 2 * action_dim)) # moyenne + log_std pour chaque action donc 2*action_dim
        return nn.Sequential(*layers)

    def predict(self, observations: List[List[float]], deterministic: bool = None):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        assert obs.shape[-1] == next(self.policy_network.children()).in_features
        params = self.policy_network(obs)  # (batch, 2 * action_dim)

        if params.dim() == 1:
            params = params.unsqueeze(0)

        n = params.shape[-1] // 2 # on sépare mean et log_std
        mean, log_std = params[:, :n], params[:, n:]

        #(instabilités numériques)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        std = torch.exp(log_std)

        dist = Normal(mean, std)

        if deterministic:
            u = mean
        else:
            u = dist.rsample() 

        a_tanh = torch.tanh(u) # borne entre -1 et 1
        
        # ---- rescale depuis [-1,1] vers [low, high] dans l'espace concaténé ----
        # low/high étaient (B, A_par_bat) → on les aplatit en (A_total)
        low_flat  = self.action_low.reshape(-1)   # (A_total,)
        high_flat = self.action_high.reshape(-1)  # (A_total,)

        Bbatch = a_tanh.shape[0]
        low_b  = low_flat.unsqueeze(0).expand(Bbatch, -1)   # (batch, A_total)
        high_b = high_flat.unsqueeze(0).expand(Bbatch, -1)  # (batch, A_total)

        a_flat = low_b + 0.5 * (a_tanh + 1.0) * (high_b - low_b)   # (batch, A_total)
        a_flat = torch.max(torch.min(a_flat, high_b), low_b)       # clip

        # ---- log_prob (avec correction tanh) ----
        if not deterministic:
            eps = 1e-6
            corr = torch.log((1.0 - a_tanh.pow(2)).clamp(min=eps))     # (batch, A_total)
            log_prob_per_dim = dist.log_prob(u) - corr                 # (batch, A_total)
            log_prob_per_sample = log_prob_per_dim.sum(dim=-1)         # (batch,)

            # Ici on suppose batch=1 (cas CityLearn classique). On garde le gradient !
            step_log_prob = log_prob_per_sample.sum()
            self._traj_log_probs.append(step_log_prob)  # <-- pas de .detach()

        # ---- re-split par bâtiment puis renvoi au format CityLearn ----
        # On suppose batch=1; si batch>1, adapter en conséquence.
        a0 = a_flat[0]                                  # (A_total,)
        chunks = torch.split(a0, self._action_sizes)    # tuple de (A_b1,), (A_b2,), ...
        actions = [c.detach().cpu().numpy().tolist() for c in chunks]
        return actions


    def update_episode(self, rewards, log_probs) -> float:
        device = self.device
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = float(r) + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device) # somme pondérée des récompenses futures
        if returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        log_probs = torch.stack(log_probs).to(device)
        loss = -(log_probs * returns.detach()).sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())
    
    # focntion update compaptible avec CityLearn
    def update(self,observations,actions,rewards,next_observations,terminated=None,truncated=None,**kwargs):
        if isinstance(rewards, (list, tuple)):
            step_reward = float(sum(rewards)) # somme des récompenses de tous les bâtiments (uniquement pour un agent centralisé)
        else:
            step_reward = float(rewards)
        self._traj_rewards.append(step_reward)

        def _to_bool(x):
            if isinstance(x, (list, tuple)):
                return any(bool(v) for v in x)
            return bool(x) if x is not None else False

        done = _to_bool(terminated) or _to_bool(truncated)
        if not done:
            return {}

        T = min(len(self._traj_rewards), len(self._traj_log_probs))
        rewards_ep = self._traj_rewards[:T]
        log_probs_ep = self._traj_log_probs[:T] # aligner les longueurs

        info = {}
        if T > 0:
            loss = self.update_episode(rewards_ep, log_probs_ep)
            info["loss"] = float(loss)

        self._traj_rewards.clear()
        self._traj_log_probs.clear()
        return info
    
    # helper pour récupérer les bornes par batiment
    def _compute_action_bounds(self, env):
        space = getattr(env, "action_space", None)

        if hasattr(space, "low") and hasattr(space, "high"):
            low  = torch.as_tensor(space.low,  dtype=torch.float32, device=self.device)
            high = torch.as_tensor(space.high, dtype=torch.float32, device=self.device)
            if low.dim() == 1:
                low  = low.unsqueeze(0)
                high = high.unsqueeze(0)
            return low, high

        # un par building
        if isinstance(space, (list, tuple)) and len(space) > 0:
            lows, highs = [], []
            buildings = getattr(env, "buildings", [])
            if buildings and hasattr(buildings[0], "estimate_action_space"):
                for b in buildings:
                    box = b.estimate_action_space()
                    lows.append(torch.as_tensor(box.low,  dtype=torch.float32, device=self.device).view(-1))
                    highs.append(torch.as_tensor(box.high, dtype=torch.float32, device=self.device).view(-1))
            else:
                for box in space:
                    lows.append(torch.as_tensor(box.low,  dtype=torch.float32, device=self.device).view(-1))
                    highs.append(torch.as_tensor(box.high, dtype=torch.float32, device=self.device).view(-1))

            low  = torch.stack(lows,  dim=0)  # (B, A)
            high = torch.stack(highs, dim=0)  # (B, A)
            return low, high

        raise ValueError("Impossible de déterminer les bornes de l'espace d'action.")