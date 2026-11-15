import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd
import torchsde

from utils.networks import MLP
from utils.initializer import initialize_weights
from utils.transformations import create_symexp_bins, twohot, expected_value, log_prob, symlog
class ControlledNSDE(nn.Module):
    sde_type = 'ito'
    noise_type = 'diagonal'

    def __init__(
            self, 
            config,
            obs_dim,
            action_dim,
            latent_dim_deter=32,
            latent_dim_stoch=32,
            context_dim=32,
            dt=0.05,
            num_bins=255, 
            bin_limit=20, 
            use_stoch_policy=True
        ):
        super().__init__()

        # split config into components
        self.config = config
        self.sde_config = config['controlled_nsde']
        self.decoder_config = config['decoder']
        self.encoder_config = config['encoder']
        self.reward_config = config['reward_head']

        self.num_bins = num_bins
        self.bin_limit = bin_limit

        # decoder obs distribution std bounds
        self.obs_min_std = 0.01
        self.obs_max_std = 1.0

        self.register_buffer('bins', create_symexp_bins(self.num_bins, self.bin_limit, device='cpu'))

        # Dimensions & other parameters
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.freebits = self.config.get("freebits", 1.0)
        self.adjoint = self.config.get("adjoint", False)
        self.method = self.config.get("method", "euler")
        self.dt = dt
        self.latent_dim_deter = latent_dim_deter
        self.latent_dim_stoch = latent_dim_stoch
        self.latent_dim = self.latent_dim_deter + self.latent_dim_stoch
        self.context_dim = context_dim

        self.use_stoch_policy = use_stoch_policy


        self.correction_mode = self.config.get("correction_mode", "learnable")
        assert self.correction_mode in ("none", "fixed", "learnable"), self.correction_mode

        if self.correction_mode == "fixed":
            self.register_buffer('log_tau_deter', torch.zeros(latent_dim_deter))
            self.register_buffer('log_tau_stoch', torch.zeros(latent_dim_stoch))
        elif self.correction_mode == "learnable":
            self.log_tau_deter = nn.Parameter(torch.zeros(latent_dim_deter))
            self.log_tau_stoch = nn.Parameter(torch.full((latent_dim_stoch,), float(torch.log(torch.tensor(4 * dt)))))


        # Networks
        self.decoder = MLP(
            config=self.decoder_config,
            input_dim=self.latent_dim,
            output_dim=2 * self.obs_dim
        )

        self.encoder = MLP(
            config=self.encoder_config,
            input_dim=self.obs_dim,
            output_dim=self.context_dim
        )   

        self.reward_head = MLP(
            config=self.reward_config,
            input_dim=self.latent_dim,
            output_dim=self.num_bins
        )
        
        # NSDE Networks
        self.init_network = MLP(
            config=self.sde_config,
            input_dim=self.context_dim,
            output_dim=self.latent_dim
        )

        self.deter_drift = MLP(
            config=self.sde_config,
            input_dim=self.latent_dim + self.action_dim,
            output_dim=self.latent_dim_deter
        )

        self.prior_drift = MLP(
            config=self.sde_config,
            input_dim=self.latent_dim_deter,
            output_dim=self.latent_dim_stoch
        )

        self.posterior_drift = MLP(
            config=self.sde_config,
            input_dim=self.latent_dim_deter + self.context_dim,
            output_dim=self.latent_dim_stoch
        )

        # add sigmoid output activation to ensure positivity of diffusion
        self.pre_diffusion = MLP(
            config=self.sde_config,
            input_dim=self.latent_dim_deter,
            output_dim=self.latent_dim_stoch
        )

        self.diffusion = nn.Sequential(
            self.pre_diffusion,
            nn.Sigmoid()
        )

        # Initialize context  

        self.embeddings = None
        self.actions = None
        self.time_steps = None

        # Imagination mode
        self.imagine = False
        self.actor = None
        self._imag_actions = []


        # SDE integration kwargs
        self.wm_kwargs = dict(
            method=self.method,
            dt=self.dt,
            adaptive=False,
            logqp=True,
            names={'drift': 'f', 'diffusion': 'g'}
        )

        self.imag_kwargs = dict(
            method=self.method,
            dt=self.dt,
            adaptive=False,
            names={'drift': 'h', 'diffusion': 'g'}
        )
        self.encode_kwargs = dict(
            method=self.method,
            dt=self.dt,
            adaptive=False,
            names={'drift': 'f', 'diffusion': 'g'}
        )


    def _deter_update(self, raw, deter):
        if self.correction_mode == "none":
            return raw
        tau_deter = self.log_tau_deter.exp().clamp(0.2, 20.0)
        return (raw - deter) / tau_deter

    def _stoch_update(self, raw, stoch):
        if self.correction_mode == "none":
            return raw
        tau_stoch = self.log_tau_stoch.exp().clamp(0.05, 10.0)
        return (raw - stoch) / tau_stoch

    def _time_index(self, time):
        return torch.searchsorted(self.time_steps, time + 0.5 * self.dt, right=True).clamp(min=1, max=self.time_steps.size(0) - 1)

    #PRIOR
    def h(self, time, state):

        deter, stoch = torch.split(state, [self.latent_dim_deter, self.latent_dim_stoch], dim=-1)

        if not self.imagine:
            time_idx = self._time_index(time)
            action = self.actions[time_idx - 1]
        else:
            action = self.actor(state.detach(), self.use_stoch_policy)
            self._imag_actions.append(action)

        next_deter = self._deter_update(self.deter_drift(torch.cat([deter, stoch, action], dim=-1)), deter) 
        deter_lookahead = deter + next_deter * self.dt
        next_stoch = self._stoch_update(self.prior_drift(deter_lookahead), stoch)

        return torch.cat([next_deter, next_stoch], dim=-1)

    #POSTERIOR
    def f(self, time, state):
        deter, stoch = torch.split(state, [self.latent_dim_deter, self.latent_dim_stoch], dim=-1)

        # action at left bound and context at right bound
        time_idx = self._time_index(time)
        action = self.actions[time_idx - 1]
        context = self.embeddings[time_idx]
        next_deter = self._deter_update(self.deter_drift(torch.cat([deter, stoch, action], dim=-1)), deter)
        deter_lookahead = deter + next_deter * self.dt
        next_stoch = self._stoch_update(self.posterior_drift(torch.cat([deter_lookahead, context], dim=-1)), stoch)

        return torch.cat([next_deter, next_stoch], dim=-1)

    
    def g(self, time, state):

        deter, stoch = torch.split(state, [self.latent_dim_deter, self.latent_dim_stoch], dim=-1)
        # under euler, torchsde calls g right after h at the same (t, y), so the last imagined action is the current one
        action = self._imag_actions[-1] if self.imagine else self.actions[self._time_index(time) - 1]
        next_deter = self._deter_update(self.deter_drift(torch.cat([deter, stoch, action], dim=-1)), deter)
        deter_lookahead = deter + next_deter * self.dt
        next_stoch = self.diffusion(deter_lookahead)

        return torch.cat([torch.full_like(deter, 1e-6), next_stoch], dim=-1)
    


    def train_world_model(
            self,
            data
        ):
        
        self.embeddings = self.encoder(symlog(data['observations'])).transpose(0, 1) # (time, batch, context_dim) for sdeint compatibility
        self.actions = data['actions'].transpose(0, 1) # (time, batch, action_dim) for sdeint compatibility
        ar = data['action_repeat'].clone()
        ar[0] = 0 # reset start index to zero for first step. 
        self.time_steps = torch.cumsum(ar, dim=0) * self.dt # (time,)


        initial_state = self.init_network(self.embeddings[0])

        sdeint_fn = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint

        post_traj, path_kl = sdeint_fn(
            self, initial_state, self.time_steps, **self.wm_kwargs
        )

        
        kl_loss = torch.clamp(path_kl, min=self.freebits).mean()

        post_traj = post_traj.transpose(0, 1) # (batch, time, latent_dim)

        obs_dist = self.get_obs_dist(post_traj)              # (B, time, 2 * obs_dim)
        # remove initial step from reward prediction
        reward_logits = self.reward_head(post_traj[:, 1:]) # (B, time-1, reward_bins)

        obs_loss = -obs_dist.log_prob(symlog(data['observations'])).mean()  # sum over obs_dim, mean over batch and time

        # rewards are both now (B, time-1)
        reward_loss = - log_prob(reward_logits, twohot(data['rewards'], self.bins).detach()).mean()

        total_loss = obs_loss + reward_loss + kl_loss

        with torch.no_grad():
            # last step has no outgoing action, so evaluate g on the first time-1 states
            deter_traj, stoch_traj = torch.split(post_traj[:, :-1], [self.latent_dim_deter, self.latent_dim_stoch], dim=-1)
            next_deter = self._deter_update(self.deter_drift(torch.cat([deter_traj, stoch_traj, data['actions']], dim=-1)), deter_traj)
            deter_lookahead = deter_traj + next_deter * self.dt
            diffusion_g = self.diffusion(deter_lookahead)  # (B, time-1, latent_dim_stoch)

            
        metrics = {
            'obs_loss': obs_loss.item(),
            'reward_loss': reward_loss.item(),
            'kl_loss': kl_loss.item(),
            'path_kl': path_kl.mean().item(),
            'freebits_fraction': (path_kl < self.freebits).float().mean().item(),
            'diffusion_mean': diffusion_g.mean().item(),
            'diffusion_max': diffusion_g.max().item(),
            'obs_dist_mean': obs_dist.mean.mean().item(),
            'obs_dist_std': obs_dist.stddev.mean().item(),
        }
        if self.correction_mode != "none":
            metrics['tau_deter_mean'] = self.log_tau_deter.exp().clamp(0.2, 20.0).mean().item()
            metrics['tau_stoch_mean'] = self.log_tau_stoch.exp().clamp(0.05, 10.0).mean().item()

        return post_traj, total_loss, metrics
    
    def imagine_trajectory(self, initial_states, policy, horizon):
        self.imagine = True
        self.actor = policy
        self._imag_actions = []

        time_steps = torch.arange(horizon, device=initial_states.device) * self.dt

        traj = torchsde.sdeint(
            self, initial_states, time_steps, **self.imag_kwargs
        )

        traj = traj.transpose(0, 1)  # (batch, horizon, latent_dim)
        actions = torch.stack(self._imag_actions, dim=1)  # (batch, horizon-1, action_dim)

        self.imagine = False
        self.actor = None
        self._imag_actions = []

        return traj, actions
    

    
    def encode_observation(self, obs, prev_action, latent_state, prev_action_repeat=None):
        obs = self.encoder(symlog(obs))
        if latent_state is None:
            initial_state = self.init_network(obs)
            self.embeddings = None

            return initial_state
        else:
            self.embeddings = torch.stack([obs, obs], dim=0) # (time=2, batch, context_dim) expected by f
            self.actions = torch.stack([prev_action, prev_action], dim=0) # (time=2, batch, action_dim) expected by f
            self.time_steps = torch.tensor([0.0, prev_action_repeat * self.dt]).to(obs.device)
            sdeint_fn = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint

            post_traj = sdeint_fn(
                self, latent_state, self.time_steps, **self.encode_kwargs
            )


            post_traj = post_traj.transpose(0, 1) # (batch, time, latent_dim)

            return post_traj[:, -1]

    def get_obs_dist(self, latent_state):
        stats = self.decoder(latent_state)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = (self.obs_max_std - self.obs_min_std) * F.sigmoid(std + 2.0) + self.obs_min_std
        return torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)