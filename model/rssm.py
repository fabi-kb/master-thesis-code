import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd


from utils.networks import MLP, GRUCell
from utils.initializer import initialize_weights
from utils.transformations import create_symexp_bins, twohot, log_prob, symlog



class RSSM(nn.Module):
    
    def __init__(
            self, 
            config,
            obs_dim,
            action_dim,
            latent_dim_deter=32,
            latent_dim_stoch=32,
            context_dim=32,
            dt=0.05, # dummy input for compatibility
            num_bins=255,
            bin_limit=20, 
            use_stoch_policy=True
        ):
        

        super().__init__()

        # split config into components
        self.config = config
        self.rssm_config = config['rssm']
        self.decoder_config = config['decoder']
        self.encoder_config = config['encoder']
        self.reward_config = config['reward_head']

        self.num_bins = num_bins
        self.bin_limit = bin_limit

        self.min_std = 0.1

        # decoder obs distribution std bounds
        self.obs_min_std = 0.01
        self.obs_max_std = 1.0

        self.use_stoch_policy = use_stoch_policy

        self.register_buffer('bins', create_symexp_bins(self.num_bins, self.bin_limit, device='cpu'))

        # Dimensions
        self.hidden_dim = self.rssm_config["hidden_dim"]
        self.latent_dim_deter = latent_dim_deter
        self.latent_dim_stoch = latent_dim_stoch
        self.context_dim = context_dim
        self.latent_dim = self.latent_dim_deter + self.latent_dim_stoch

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Networks
        self.decoder = MLP(
            config=self.decoder_config,
            input_dim=latent_dim_deter + latent_dim_stoch,
            output_dim=2 * self.obs_dim
        )

        self.encoder = MLP(
            config=self.encoder_config,
            input_dim=self.obs_dim,
            output_dim=self.context_dim
        )

        self.reward_head = MLP(
            config=self.reward_config,
            input_dim=self.latent_dim_deter + self.latent_dim_stoch,
            output_dim=self.num_bins
        )

        # RSSM-specific networks

        self.img_in = MLP(
            config=self.rssm_config,
            input_dim=self.latent_dim_stoch + self.action_dim,
            output_dim=self.hidden_dim
        )

        self.img_out = MLP(
            config=self.rssm_config,
            input_dim=self.latent_dim_deter,
            output_dim=2 * self.latent_dim_stoch
        )


        self.obs_out = MLP(
            config=self.rssm_config,
            input_dim=self.latent_dim_deter + self.context_dim,
            output_dim=2 * self.latent_dim_stoch
        )

        self.initial_deter = torch.nn.Parameter(torch.zeros((1, self.latent_dim_deter), requires_grad=True))


        self.cell = GRUCell(self.hidden_dim, self.latent_dim_deter, norm=True)





    def train_world_model(self, data):
        observations = data['observations']  # (B, T, obs_dim)
        actions = data['actions']            # (B, T-1, action_dim)
        rewards = data['rewards']            # (B, T-1)

        B, T, _ = observations.shape
        freebits = self.config.get("freebits", 1.0)
        context = self.encoder(symlog(observations))

        post_states = []
        dyn_losses = []
        rep_losses = []
        state = self.initial(B).to(observations.device)

        for t in range(T):
            prev_action = actions[:, t - 1] if t > 0 else torch.zeros(B, self.action_dim, device=observations.device)
            context_t = context[:, t]

            post_state, prior_state = self.obs_step(state, prev_action, context_t)

            prior_deter = prior_state[:, :self.latent_dim_deter]
            prior_dist = self.get_dist(prior_deter, img_mode=True)
            post_dist = self.get_dist(torch.cat([prior_deter, context_t], dim=-1), img_mode=False)
            dyn_loss, rep_loss = self.kl_loss_function(post_dist, prior_dist, freebits=freebits)

            dyn_losses.append(dyn_loss)
            rep_losses.append(rep_loss)
            post_states.append(post_state)
            state = post_state

        post_traj = torch.stack(post_states, dim=1)  # (B, T, latent_dim)

        # mean over timesteps and batch to get scalars
        dyn_loss = torch.stack(dyn_losses, dim=1).mean()
        rep_loss = torch.stack(rep_losses, dim=1).mean()

        obs_dist = self.get_obs_dist(post_traj)              # (B, T, 2 * obs_dim)
        reward_logits = self.reward_head(post_traj[:, 1:])  # (B, T-1, num_bins)

        obs_loss = -obs_dist.log_prob(symlog(observations)).mean()  # sum over obs_dim, mean over batch and time

        reward_loss = -log_prob(reward_logits, twohot(rewards, self.bins).detach()).mean()
        total_loss = obs_loss + reward_loss + 0.5 * dyn_loss + 0.1 * rep_loss 

        return post_traj, total_loss, {
            'obs_loss': obs_loss.item(),
            'reward_loss': reward_loss.item(),
            'dyn_loss': dyn_loss.item(),
            'rep_loss': rep_loss.item(),
            'obs_dist_mean': obs_dist.mean.mean().item(),
            'obs_dist_std': obs_dist.stddev.mean().item()
        }


    def imagine_trajectory(self, initial_states, policy, horizon):
        traj = [initial_states]
        state = initial_states

        actions = []

        for _ in range(horizon - 1):
            action = policy(state.detach(), self.use_stoch_policy)
            state = self.img_step(state, action)
            traj.append(state)
            actions.append(action)

        return torch.stack(traj, dim=1), torch.stack(actions, dim=1)  # (B, horizon, latent_dim), (B, horizon-1, action_dim)


    def encode_observation(self, obs, prev_action, latent_state, prev_action_repeat=None): # dummy input for compatibility
        if latent_state is None:
            latent_state = self.initial(obs.size(0)).to(obs.device)
            prev_action = torch.zeros(obs.size(0), self.action_dim, device=obs.device)

        post_state, _ = self.obs_step(latent_state, prev_action, self.encoder(symlog(obs)), sample=False)
        return post_state


    ### Helper functions
    def get_dist(self, deter, img_mode=True):
        # multivariate normal distribution with mean and std given by the img_out network
        if img_mode:
            stats = self.img_out(deter)
        else:
            stats = self.obs_out(deter)

        mean, std = torch.chunk(stats, 2, dim=-1)
        std = 2 * F.sigmoid(std/2) + self.min_std

        return torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)

    def initial(self, batch_size):
        deter = torch.tanh(self.initial_deter).expand(batch_size, -1)
        stoch = self.get_dist(deter).mean
        return torch.cat([deter, stoch], dim=-1)
    
    def obs_step(self, prev_state, prev_action, obs, sample=True):
        if prev_state is None:
            return self.initial(obs.size(0)), torch.zeros((obs.size(0), self.action_dim), device=obs.device)
        
        prior = self.img_step(prev_state, prev_action)
        # concatenate prior deter with obs and pass through obs_out to get posterior distribution
        x = torch.cat([prior[:, :self.latent_dim_deter], obs], dim=-1)

        post_stoch = self.get_dist(x, img_mode=False).rsample() if sample else self.get_dist(x, img_mode=False).mean  # (B, latent_dim_stoch)

        return torch.cat([prior[:, :self.latent_dim_deter], post_stoch], dim=-1), prior

    def img_step(self, prev_state, prev_action, sample=True):

        prev_deter = prev_state[:, :self.latent_dim_deter]
        prev_stoch = prev_state[:, self.latent_dim_deter:]


        x = torch.cat([prev_stoch, prev_action], dim=-1)
        x = self.img_in(x) # -> (B, hidden_dim)

        x, deter = self.cell(x, prev_deter) # -> (B, latent_dim_deter), (B, latent_dim_deter)

        stoch = self.get_dist(x, img_mode=True).rsample() if sample else self.get_dist(x).mean  # -> (B, latent_dim_stoch)

        return torch.cat([deter, stoch], dim=-1)



    def kl_loss_function(self, posterior_dist, prior_dist, freebits=1.0):
        kld = torchd.kl.kl_divergence

        # The DreamerV3 reference implementation passes param dicts and does: sg = lambda x: {k: v.detach() for k, v in x.items()}
        # then calls get_dist(sg(prior)). We pass distributions directly, so we reconstruct
        # from detached parameters — equivalent effect.
        sg = lambda d: torchd.independent.Independent(
            torchd.normal.Normal(d.base_dist.loc.detach(), d.base_dist.scale.detach()), 1)

        rep_loss = kld(posterior_dist, sg(prior_dist))   # KL(post || sg(prior)) — trains posterior
        dyn_loss = kld(sg(posterior_dist), prior_dist)   # KL(sg(post) || prior) — trains prior/dynamics

        rep_loss = torch.clamp(rep_loss, min=freebits)
        dyn_loss = torch.clamp(dyn_loss, min=freebits)

        return dyn_loss, rep_loss


    def get_obs_dist(self, latent_state):
        stats = self.decoder(latent_state)
        mean, std = torch.chunk(stats, 2, dim=-1)
        std = (self.obs_max_std - self.obs_min_std) * F.sigmoid(std + 2.0) + self.obs_min_std
        return torchd.independent.Independent(torchd.normal.Normal(mean, std), 1)
