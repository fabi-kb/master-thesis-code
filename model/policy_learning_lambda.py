import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from utils.networks import MLP
from utils.initializer import initialize_weights
from utils.transformations import create_symexp_bins, twohot, expected_value, log_prob


class TwoHeadedActorMLP(nn.Module):

    def __init__(self, config, input_dim, output_dim):
        super().__init__()

        hidden_dim = config['hidden_dim']
        hidden_layers = config['hidden_layers']
        bias = config.get('bias', True)
        activation = getattr(nn, config['activation'])
        layer_norm_cls = getattr(nn, config['layer_norm']) if config['layer_norm'] != 'None' else None

        self._absmax = config.get('absmax', 1.0)
        self._min_std = config.get('min_std', 0.1)
        self._max_std = config.get('max_std', 1.0)

        layers = []
        for i in range(hidden_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if layer_norm_cls is not None:
                layers.append(layer_norm_cls(hidden_dim))
            layers.append(activation())

        self.hidden = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.std_head = nn.Linear(hidden_dim, output_dim, bias=bias)

        initialize_weights(self.hidden, config['weight_dist'], config['weight_dist'], config['fan_mode'])
        initialize_weights(nn.ModuleList([self.mean_head]), config['weight_dist'], config['output_weight_dist'], config['fan_mode'])
        initialize_weights(nn.ModuleList([self.std_head]), config['weight_dist'], config['output_weight_dist'], config['fan_mode'])

    def forward(self, x, sample=False):
        if sample: 
            out = self.sample(x)
        else:
            mean = self.mean_head(self.hidden(x))
            scale = (self._absmax / torch.clip(torch.abs(mean), min=self._absmax)).detach()
            out = scale * mean

        return out

    def sample(self, x):
        dist = self.distribution(x).rsample()
        out = (self._absmax / torch.clip(torch.abs(dist), min=self._absmax)).detach() * dist
        return out

    def distribution(self, x):
        h = self.hidden(x)
        mean = self.mean_head(h)
        # sigmoid with +2.0 offset starts std near max_std; range [min_std, max_std]
        std = (self._max_std - self._min_std) * torch.sigmoid(self.std_head(h) + 2.0) + self._min_std
        return torch.distributions.Normal(mean, std)

    def raw_mean(self, x):
        return self.mean_head(self.hidden(x))


class PolicyLearning(nn.Module):

    def __init__(
            self,
            config: dict,
            latent_dim_deter: int,
            latent_dim_stoch: int,
            action_dim: int,
            world_model: nn.Module,
    ):
        super().__init__()

        self.config = config
        self.world_model = world_model

        self.latent_dim_deter = latent_dim_deter
        self.latent_dim_stoch = latent_dim_stoch
        self.latent_dim = latent_dim_deter + latent_dim_stoch
        self.action_dim = action_dim

        self.num_bins = self.config.get("num_bins", 255)
        self.bin_limit = self.config.get("bin_limit", 20)

        self.register_buffer('bins', create_symexp_bins(self.num_bins, self.bin_limit, device='cpu'))


        self.horizon = config.get("horizon", 15)
        self.gamma = config.get("gamma", 0.99)
        self.lambda_ = config.get("lambda_", 0.95)  # for lambda returns

        self.norm_limit = config.get("ret_norm_limit", 1e-8)
        self.norm_rate = config.get("ret_norm_rate", 0.01)

        self.entropy_coeff = config.get('entropy_coeff', 3e-4)

        self.use_stoch_policy = config.get('use_stoch_policy', True)

        self.actor = TwoHeadedActorMLP(
            config=config['actor'],
            input_dim=self.latent_dim,
            output_dim=self.action_dim
        )

        self.critic = MLP(
            config=config['critic'],
            input_dim=self.latent_dim,
            output_dim=self.num_bins
        )

        # create slow critic
        self.slow_critic = deepcopy(self.critic)

        self.actor_grad_clip = config.get('actor_grad_clip', 100.0)
        self.critic_grad_clip = config.get('critic_grad_clip', 100.0)
        self.slow_critic_tau = config.get('slow_critic_tau', 0.02)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config['actor_lr'], eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config['critic_lr'], eps=1e-5)

        # normalizer for advantages
        self.ret_norm = self.PercentileNormalizer(rate=self.norm_rate, limit=self.norm_limit)

    def train_policy(self, initial_states):

        imag_traj, _ = self.world_model.imagine_trajectory(
            initial_states=initial_states.detach(),
            policy=self.actor,
            horizon=self.horizon
        )
        # PRECOMPUTATIONS
        imag_traj_detached = imag_traj.detach()

        reward_logits = self.world_model.reward_head(imag_traj)
        reward_hat = expected_value(reward_logits, self.bins)

        values_logits = self.critic(imag_traj)
        values = expected_value(values_logits, self.bins)

        lambda_rets = self.compute_lambda_returns(reward_hat, values)


        # ACTOR LOSS
        ret_offset, ret_scale = self.ret_norm(lambda_rets, update=True)

        advantage = lambda_rets - values[:, :-1]

        norm_advantage = advantage / ret_scale

        actor_dist = self.actor.distribution(imag_traj)
        entropy = actor_dist.entropy()

        actor_loss = -norm_advantage.mean() - self.entropy_coeff * entropy.mean()

        # CRITIC LOSS 

        value_logits = self.critic(imag_traj_detached[:, :-1])
        value_loss = - log_prob(value_logits, twohot(lambda_rets, self.bins).detach())



        slow_values_logits = self.slow_critic(imag_traj_detached[:, :-1])
        slow_values = expected_value(slow_values_logits, self.bins)
        slow_nll = log_prob(value_logits, twohot(slow_values, self.bins).detach())
        value_loss -= slow_nll 

        critic_loss = value_loss.mean()
        
        # update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_optimizer.step()

        # update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clip)
        self.critic_optimizer.step()

        self.update_slow_networks(tau=self.slow_critic_tau)

        # for logging 
        ret_normed = (lambda_rets - ret_offset) / ret_scale

        with torch.no_grad():
            current_actions = self.actor(imag_traj_detached)
            actor_raw_mean = self.actor.raw_mean(imag_traj_detached)
            log_entropy = self.actor.distribution(imag_traj_detached).entropy()
            metrics = {
                'val_pred_mean': values[:, :-1].mean().item(),
                'val_pred_max': values[:, :-1].max().item(),
                'val_pred_min': values[:, :-1].min().item(),
                'targets_mean': lambda_rets.mean().item(),
                'targets_max': lambda_rets.max().item(),
                'targets_min': lambda_rets.min().item(),
                'reward_mean': reward_hat.mean().item(),
                'action_mean': current_actions.mean().item(),
                'actor_raw_mean': actor_raw_mean.mean().item(),
                'latent_norm': imag_traj_detached.norm(dim=-1).mean().item(),
                'advantage_mean': advantage.mean().item(),
                'advantage_std': advantage.std().item(),
                'advantage_normed_mean': norm_advantage.mean().item(),
                'target_norm_min': ret_normed.min().item(),
                'target_norm_max': ret_normed.max().item(),
                'ret_offset': ret_offset.item(),
                'ret_scale': ret_scale.item(),
                'actor_grad_norm': actor_grad_norm.item(),
                'critic_grad_norm': critic_grad_norm.item(),
                'entropy_mean': log_entropy.mean().item(),
            }

        return actor_loss, critic_loss, metrics
    
    def update_slow_networks(self, tau):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(tau * param.data + (1.0 - tau) * slow_param.data)

    def compute_lambda_returns(self, rewards, values):    
        lam_rets = [values[..., -1]]

        intermediate = rewards[..., 1:] + (1- self.lambda_) * self.gamma * values[..., 1:]

        for t in reversed(range(intermediate.size(-1))):
            lam_rets.append(intermediate[..., t] + self.lambda_ * self.gamma * lam_rets[-1])
        
        return torch.stack(list(reversed(lam_rets))[:-1], dim=-1)


    class PercentileNormalizer(nn.Module):

        def __init__(self, rate: float = 0.01, limit: float = 1.0):
            super().__init__()

            self.rate = rate
            self.limit = limit
            # ema_vals[0] = EMA of p05, ema_vals[1] = EMA of p95
            self.register_buffer('ema_vals', torch.zeros(2))

        def forward(self, x, update=True):
            if update:
                flat_x = x.detach().float().flatten()
                quantiles = torch.quantile(
                    flat_x,
                    torch.tensor([0.05, 0.95], device=flat_x.device, dtype=flat_x.dtype)
                )
                self.ema_vals.copy_(self.rate * quantiles + (1 - self.rate) * self.ema_vals)

            offset = self.ema_vals[0].clone()
            scale = torch.clamp(self.ema_vals[1] - self.ema_vals[0], min=self.limit)
            return offset.detach(), scale.detach()