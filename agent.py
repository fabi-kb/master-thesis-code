import torch
import torchsde
import yaml
import gymnasium as gym

from pathlib import Path

from typing import Dict, Any
from tqdm import tqdm



# internal imports
from environments.dmc_environment import DMCEnvironment
from environments.gym_environment import GymEnvironment

from model.controlled_nsde import ControlledNSDE
from model.rssm import RSSM
from model.policy_learning_lambda import PolicyLearning

from utils.logger import Logger
from utils.replay_buffer_irregular import BatchReplayBuffer
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.plotting import plot_evaluation
from utils.transformations import symexp 


import numpy as np

class DreamerAgent():
    

    def __init__(self, config_dict: dict = None, log_dir: str = "runs"):
        # loading logic for config as new model or from checkpoint with dict
        if config_dict is not None:
            self.config = config_dict
        else:
            raise ValueError("config_dict must be provided.")

        self.log_dir = Path(log_dir)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # split config into components
        self.env_config = self.config['env']
        self.train_config = self.config['train']
        self.model_config = self.config['model']
        self.policy_config = self.config['policylearning']


        # -------------- ENVIRONMENT --------------
        # list of all environments currently supported 
        env_classes = {
            'dmc': DMCEnvironment,
            'gym': GymEnvironment
        }

        env_class = env_classes.get(self.env_config['backend'], GymEnvironment)
        
        env_fns = [lambda i=i: env_class(
            name=self.env_config['name'],
            time_limit=self.env_config.get('time_limit', None),
            dt=self.env_config.get('dt', None),
            seed=self.env_config.get('seed', 0) * 100 + i,
            action_repeat=self.env_config.get('action_repeat', 1)
        ) for i in range(self.train_config.get("num_envs", 1))]

        self.envs = gym.vector.AsyncVectorEnv(env_fns)

        self.action_low = self.envs.action_space.low[0]
        self.action_high = self.envs.action_space.high[0]
        print(f"Action space low: {self.action_low}, high: {self.action_high}")

        # -------------- REPLAY BUFFER --------------

        self.batch_size = self.train_config.get("batch_size", 16)
        self.time_limit = self.env_config.get("time_limit", 1000)

        self.replay_buffer = BatchReplayBuffer(
            capacity=self.train_config.get("replay_buffer_capacity", 10000),
            time_limit=self.time_limit,
            observation_shape=self.envs.observation_space.shape[1:], # first dim is batch size
            action_shape=self.envs.action_space.shape[1:], # first dim is batch size
            batch_size=self.batch_size
        )
        

        # -------------- WORLD MODEL --------------

        self.obs_dim = self.envs.observation_space.shape[1] # first dimension is batch size, second is obs dim
        self.action_dim = self.envs.action_space.shape[1]
        print(f"Observation space: {self.obs_dim}, Action space: {self.action_dim}")
        self.sequence_length = self.train_config.get("sequence_length", 20)
        self.dt = self.env_config.get("dt", 0.05) * self.env_config.get("action_repeat", 1) # physical dt = env dt * action repeat

        # latent dimensions
        self.latent_dim_deter = self.model_config.get("latent_dim_deter", 32)
        self.latent_dim_stoch = self.model_config.get("latent_dim_stoch", 32)
        self.latent_dim = self.model_config.get("latent_dim", self.latent_dim_deter + self.latent_dim_stoch)
        self.context_dim = self.model_config.get("context_dim", 32)

        world_models = {
            'controlled_nsde': ControlledNSDE,
            'rssm': RSSM
        }

        world_model_class = world_models.get(self.model_config['name'], ControlledNSDE)


        self.world_model = world_model_class(
            config=self.model_config, 
            obs_dim=self.obs_dim, 
            action_dim=self.action_dim,
            latent_dim_deter=self.latent_dim_deter,
            latent_dim_stoch=self.latent_dim_stoch,
            context_dim=self.context_dim,
            dt=self.dt,
            num_bins=self.policy_config.get("num_bins", 255),
            bin_limit=self.policy_config.get("bin_limit", 20),
            use_stoch_policy=self.policy_config.get("use_stoch_policy", True)
        ).to(self.device)
        
        
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.model_config.get("learning_rate", 1e-3))

        print(f"World model parameters: {sum(p.numel() for p in self.world_model.parameters())}")
        print(f"World model learning rate: {self.model_config.get('learning_rate', 1e-3)}")


        # -------------- POLICY LEARNING --------------
        
        
        self.policy_learning = PolicyLearning(
            config=self.policy_config,
            latent_dim_deter=self.latent_dim_deter,
            latent_dim_stoch=self.latent_dim_stoch,
            action_dim=self.action_dim,
            world_model=self.world_model
        ).to(self.device)

        self.use_stoch_policy = self.policy_config.get("use_stoch_policy", True)


        # -------------- EVALUATION --------------

        eval_env_fns = [lambda: env_class(
            name=self.env_config['name'],
            time_limit=self.env_config.get('time_limit', None),
            dt=self.env_config.get('dt', None),
            action_repeat=self.env_config.get('action_repeat', 1)
        ) for _ in range(self.train_config.get("eval_num_envs", 1))]

        self.eval_envs = gym.vector.AsyncVectorEnv(eval_env_fns)
        


        # -------------- LOGGING --------------
        self.logger = Logger(log_dir=self.log_dir)


        # -------------- CHECKPOINTING --------------
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # -------------- TRAINING ---------------
        self.train_steps = self.train_config.get("train_steps", 100000)

        
        self.global_step = 0

        # -------------- IRREGULAR SAMPLING --------------

        self.irregular_sampling = self.env_config.get("irregular_sampling", False)
        self.irregular_max_repeat = self.env_config.get("irregular_max_repeat", 5)
        self.irregular_dist = self.env_config.get("irregular_dist", "uniform") # At this state only uniform allowed



        # -------------- HELPERS FOR TRAINING --------------
        self.current_latent_state = None
        self.prev_action = None
        self.prev_action_repeat = 1
        self.episode_rewards = np.zeros(self.envs.num_envs) # for tracking episode rewards across parallel envs
        self.train_updates = int(max(1, self.envs.num_envs * self.train_config.get("train_ratio", 0.5)))


    def train(self):
        
        # placeholder until the first policy update
        imag_metrics = {}

        # Prefill replay buffer

        timestep = 0

        print("Prefill replay buffer...")

        timestep, next_obs = self.prefill_replay_buffer(timestep=timestep)        

        print("Starting training loop...")

        pbar = tqdm(total=self.train_steps, desc="Training", initial=self.global_step)
        
        # count environment steps in increments of batch size since we step all envs in parallel
        while self.global_step < self.train_steps:

            steps_taken = self.envs.num_envs 
            self.global_step += steps_taken
            pbar.update(steps_taken)

            # do environment step and add to replay buffer
            timestep, next_obs = self.step_environment(timestep=timestep, observation=next_obs)

            for _ in range(self.train_updates):

                # sample batch from replay buffer
                batch_data = self.replay_buffer.sample_batch(sequence_length=self.sequence_length, device=self.device)

                # update world model
                post_traj, wm_loss, metrics = self.world_model.train_world_model(batch_data)
                
                self.wm_optimizer.zero_grad()
                wm_loss.backward()
                self.wm_optimizer.step()

                if self.global_step >= self.train_config.get("pretrain_steps", 0):
                    # update policy
                    # stack post_traj from (B, time, latent_dim) to (B*time, latent_dim) for policy training
                    B, T, D = post_traj.shape
                    post_traj_flat = post_traj.detach().reshape(B * T, D)
                    actor_loss, critic_loss, imag_metrics = self.policy_learning.train_policy(post_traj_flat)

                    # add policy losses to metrics for logging
                    metrics['actor_loss'] = actor_loss.item()
                    metrics['critic_loss'] = critic_loss.item()

            # log metrics
            if self.global_step % self.train_config.get("log_every", 10) < steps_taken:
                self.logger.log_training_metrics(metrics, self.global_step)
                if self.global_step >= self.train_config.get("pretrain_steps", 0):
                    self.logger.log_imagination_metrics(imag_metrics, self.global_step)
            
            if self.global_step % self.train_config.get("print_every", 200) < steps_taken:
                print(f"Step {self.global_step}, World Model Loss: {wm_loss.item():.4f}, Metrics: {metrics}")


            if self.global_step % self.train_config.get("checkpoint_every", 1000) < steps_taken and self.global_step > 0:
                # save model checkpoint
                save_checkpoint(self, path=self.checkpoint_dir, name=f'checkpoint_{self.global_step}.pth')

            if self.global_step % self.train_config.get("eval_every", 1000) < steps_taken:
                # evaluate policy
                eval_metrics = self.evaluate(num_episodes=self.train_config.get("eval_episodes", 10))
                self.logger.log_evaluation_metrics(eval_metrics, self.global_step)
            

        # save final checkpoint at the end of training
        save_checkpoint(self, path=self.checkpoint_dir, name=f'final_model.pth')



    def prefill_replay_buffer(self, timestep=0):
        
        # add seed here to have deterministic prefill for gym, since it does not support seeding in __init__ like dmc
        obs, _ = self.envs.reset(seed=self.env_config.get("seed", None)) 

        self.replay_buffer.add_observation_batch(
            action_repeat=0, # initial timestep is zero
            observation=obs,
            action=None, # no action at first step
            reward=None, # no reward at first step
            first_step=True
            )

        for _ in tqdm(range(self.sequence_length), desc="Prefilling replay buffer"):

            action = self.envs.action_space.sample() # random for prefill

            if self.irregular_sampling:

                repeat_val = np.random.randint(1, self.irregular_max_repeat + 1, size=1).item() # random integer between 1 and irregular_max_repeat

                for _repeat in range(repeat_val):
                    next_obs, reward, terminated, limit_reached, info = self.envs.step(action)
                    if limit_reached.any():
                        repeat_val = _repeat + 1 # update repeat_val to the number of steps actually taken before the limit
                        break # break if any environment reaches its limit

                action_repeat = repeat_val

            else: 
                next_obs, reward, terminated, limit_reached, info = self.envs.step(action)

                action_repeat = 1 # default action repeat for non-irregular sampling

            action_scaled = self.downscale_action(action) 
            # add to replay buffer
            self.replay_buffer.add_observation_batch(
                action_repeat=action_repeat, 
                observation=next_obs,
                action=action_scaled,
                reward=reward,
                first_step=False
            )

            self.episode_rewards += reward

            if limit_reached.any():

                next_obs, _ = self.envs.reset()

                self.replay_buffer.add_observation_batch(
                    action_repeat=0, # reset timestep to zero for new episode
                    observation=next_obs,
                    action=None, # no action at first step
                    reward=None, # no reward at first step
                    first_step=True
                )

                self.episode_rewards = np.zeros(self.envs.num_envs) # reset episode rewards for new episode

                timestep = 0 # reset timestep to zero for new episode
            else:
                timestep += 1

        return timestep, next_obs
            

    def step_environment(self, timestep=0, observation=None):
        
        # force re-initialization of latent state to prevent drifting dynamics
        if timestep % self.sequence_length == 0 and timestep > 0 and not self.model_config.get("energy_preserving", False):
            self.current_latent_state = None
            self.prev_action = None
            print(f"Resetting latent state after reaching sequence length limit, with current timestep: {timestep}")

        with torch.no_grad():
            # Convert incoming observation numpy array to PyTorch Tensor
            observation_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device)
            
            # Reinitialize the latent state
            if self.current_latent_state is None:
                self.current_latent_state = self.world_model.encode_observation(
                    obs=observation_tensor,
                    prev_action=None,
                    latent_state=None,
                    prev_action_repeat=None
                )

            else: 
                self.current_latent_state = self.world_model.encode_observation(
                    obs=observation_tensor,
                    prev_action=self.prev_action,
                    latent_state=self.current_latent_state,
                    prev_action_repeat=self.prev_action_repeat
                )
            

            action_mean = self.policy_learning.actor(self.current_latent_state, self.use_stoch_policy)  # get action from policy
            self.prev_action = action_mean
            action = self.prev_action.cpu().numpy() # convert action to numpy for env step
            action_upscaled = self.upscale_action(action) # upscale action to environment action space

        if self.irregular_sampling:
            repeat_val = np.random.randint(1, self.irregular_max_repeat + 1, size=1).item() # random integer between 1 and irregular_max_repeat

            for _repeat in range(repeat_val):
                next_obs, reward, terminated, limit_reached, info = self.envs.step(action_upscaled)
                if limit_reached.any():
                    repeat_val = _repeat + 1 # update repeat_val to the number of steps actually taken before the limit
                    break # break if any environment reaches its limit

            action_repeat = repeat_val
            self.prev_action_repeat = action_repeat # store the action repeat for the next step, only updated if irregular sampling is used
        else:
            next_obs, reward, terminated, limit_reached, info = self.envs.step(action_upscaled)
            action_repeat = 1 # default action repeat for non-irregular sampling


        # add to replay buffer
        self.replay_buffer.add_observation_batch(
            action_repeat=action_repeat,
            observation=next_obs,
            action=action, # store action in [-1, 1] space in replay buffer
            reward=reward,
            first_step=False
        )

        self.episode_rewards += reward

        if limit_reached.any():  # currently only continuing environments, has to be decoupled when using terminating ones

            self.current_latent_state = None # reset latent state when episode ends
            self.prev_action = None # reset prev_action when episode ends
            self.track_steps = 0 # reset track steps when episode ends

            
            next_obs, _ = self.envs.reset()

            self.replay_buffer.add_observation_batch(
                action_repeat=0, # reset timestep to zero for new episode
                observation=next_obs,
                action=None, # no action at first step
                reward=None, # no reward at first step
                first_step=True
            )

            print(f"Episode ended. Rewards: {self.episode_rewards.mean():.4f} ± {self.episode_rewards.std():.4f}")
            self.episode_rewards = np.zeros(self.envs.num_envs) # reset episode rewards for new episode

            timestep = 0 # reset timestep to zero for new episode
        else: 
            timestep += 1

        return timestep, next_obs

    def evaluate(self, num_episodes=10):
        # --- PLOTTING START ---
        _plot_prior_horizon = self.policy_learning.horizon
        _plot_dir = self.log_dir / 'eval_plots' / f'step_{self.global_step:07d}'
        _plot_episodes = num_episodes
        _do_plot = True
        # --- PLOTTING END ---

        eval_metrics = {
            'episode_rewards': [],
            'max_episode_reward': [],
            'min_episode_reward': [],
            'std_episode_reward': [],
            'posterior_mse': [],
        }

        for episode in range(num_episodes):

            obs, _ = self.eval_envs.reset()
            done = np.zeros(self.eval_envs.num_envs, dtype=bool)
            episode_rewards = np.zeros(self.eval_envs.num_envs)

            eval_latent_state = None
            eval_prev_action = None

            counter = 0

            episode_reconstructions = []
            episode_observations = [obs]
            # --- PLOTTING START ---
            episode_prior_rollouts = []   # multi-step imagination snapshots
            episode_prior_1step = []      # per-step 1-step-ahead prior predictions
            episode_reset_times = []      # times (seconds) where latent state was reset
            # --- PLOTTING END ---

            while not done.all():

                if counter % self.sequence_length == 0 and counter > 0 and not self.model_config.get("energy_preserving", False):
                # Latent reset if energy_preserving is set
                    eval_latent_state = None
                    eval_prev_action = None
                    print(f"Resetting latent state during evaluation after reaching sequence length limit, with current counter: {counter}")
                    # --- PLOTTING START ---
                    episode_reset_times.append(counter * self.dt)
                    # --- PLOTTING END ---

                with torch.no_grad():
                    observation_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

                    if eval_latent_state is None:
                        eval_latent_state = self.world_model.encode_observation(
                            obs=observation_tensor,
                            prev_action=None,
                            latent_state=None,
                            prev_action_repeat=None
                        )
                    else:
                        eval_latent_state = self.world_model.encode_observation(
                            obs=observation_tensor,
                            prev_action=eval_prev_action,
                            latent_state=eval_latent_state,
                            prev_action_repeat=1, # always use 1 because we evaluate at every step, so action repeat is always 1
                        )
                    
                    eval_recons = symexp(self.world_model.get_obs_dist(eval_latent_state).mean)
                    episode_reconstructions.append(eval_recons.cpu().numpy())

                    # --- PLOTTING START ---
                    if _do_plot and episode < _plot_episodes:
                        # 1-step-ahead prior prediction at every step
                        imag_1step, _ = self.world_model.imagine_trajectory(
                            initial_states=eval_latent_state.clone(),
                            policy=self.policy_learning.actor,
                            horizon=2,  # time_steps=[0, dt]; index 1 = 1 step ahead
                        )  # (num_envs, 2, latent_dim)
                        prior_1step_dec = symexp(self.world_model.get_obs_dist(imag_1step[:, 1]).mean).cpu().numpy()
                        episode_prior_1step.append(prior_1step_dec[0])  # (obs_dim,), env 0

                        # Multi-step rollout snapshot every sequence_length steps
                        if counter % self.sequence_length == 0:
                            imag_traj, _ = self.world_model.imagine_trajectory(
                                initial_states=eval_latent_state.clone(),
                                policy=self.policy_learning.actor,
                                horizon=_plot_prior_horizon,
                            )  # (num_envs, horizon, latent_dim)
                            prior_decoded = symexp(self.world_model.get_obs_dist(imag_traj).mean).cpu().numpy()
                            episode_prior_rollouts.append({
                                'start': counter,
                                'decoded': prior_decoded[0],  # env 0, shape (horizon, obs_dim)
                            })
                    # --- PLOTTING END ---

                    action_mean = self.policy_learning.actor(eval_latent_state)
                    eval_prev_action = action_mean
                    action = action_mean.cpu().numpy()
                    action = self.upscale_action(action)

                next_obs, reward, terminated, limit_reached, info = self.eval_envs.step(action)

                episode_rewards += reward
                done = np.logical_or(terminated, limit_reached)
                obs = next_obs
                episode_observations.append(obs)
                counter += 1

            eval_metrics['episode_rewards'].append(episode_rewards.mean())
            eval_metrics['max_episode_reward'].append(episode_rewards.max())
            eval_metrics['min_episode_reward'].append(episode_rewards.min())
            eval_metrics['std_episode_reward'].append(episode_rewards.std())
            eval_metrics['posterior_mse'].append(np.mean((np.array(episode_reconstructions) - np.array(episode_observations[:-1]))**2))

            print(f"Evaluation episode {episode+1}/{num_episodes} - Rewards: {episode_rewards.mean():.4f} ± {episode_rewards.std():.4f}")

            # --- PLOTTING START ---
            if _do_plot and episode < _plot_episodes:
                obs_arr  = np.array(episode_observations[:-1])[:, 0, :]  # (T, obs_dim), env 0
                post_arr = np.array(episode_reconstructions)[:, 0, :]    # (T, obs_dim), env 0
                p1s_arr  = np.array(episode_prior_1step)                  # (T, obs_dim), env 0
                post_path, prior_path = plot_evaluation(
                    episode_idx=episode,
                    global_step=self.global_step,
                    observations=obs_arr,
                    posterior_recons=post_arr,
                    prior_rollouts=episode_prior_rollouts,
                    prior_1step=p1s_arr,
                    reset_times=episode_reset_times,
                    save_dir=_plot_dir,
                    dt=self.dt,
                )
                print(f"  Saved plots: {post_path.name}  |  {prior_path.name}")
            # --- PLOTTING END ---

        return eval_metrics

# ----- Helper functions -----
    def upscale_action(self, action):
        # scale action from [-1, 1] to action space of environment
        scaled_action = self.action_low + (action + 1.0) * 0.5 * (self.action_high - self.action_low)
        return scaled_action
    
    def downscale_action(self, action):
        # scale action from action space of environment to [-1, 1]
        scaled_action = 2.0 * (action - self.action_low) / (self.action_high - self.action_low) - 1.0
        return scaled_action


def update_config(config, key_path, value):
    keys = key_path.split('.')
    d = config 

    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]

    target_key = keys[-1]

    if target_key in d and d[target_key] is not None:
        orig_type = type(d[target_key])
        if orig_type is bool:
            d[target_key] = value.lower() in ['true', '1', 'yes']
        else:
            d[target_key] = orig_type(value)
    else:

        try: d[target_key] = int(value)
        except ValueError:
            try: d[target_key] = float(value)
            except ValueError:
                try: d[target_key] = float(value)
                except ValueError:
                    d[target_key] = value


def main():

    import argparse
    import os
    import re

    parser = argparse.ArgumentParser(description='Train a Dreamer agent.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    parser.add_argument('--logdir', type=str, default='runs', help='Directory to save logs and models.')
    parser.add_argument('--opts', nargs='*', default=[],
                        help='Modify config option in dot notation, e.g. --opts env.name=cartpole env.time_limit=200 train.batch_size=32')


    args = parser.parse_args()
    config_dict = None
    if args.config is not None:
        with open(Path(args.config), 'r') as f:
            config_dict = yaml.safe_load(f)

    for opt in args.opts:
        if '=' in opt:
            key_path, value = opt.split('=', 1)
            update_config(config_dict, key_path, value)
            print(f"Updated config: {key_path} = {value}")

    final_logdir = Path(args.logdir)
    if args.opts:
        suffix = "_".join([f"{opt.split('=', 1)[0].split('.')[-1]}_{opt.split('=', 1)[1]}" for opt in args.opts if '=' in opt])
        final_logdir = final_logdir / suffix[:100]  # limit suffix length to avoid filesystem issues


    # Loading logic for checkpoint - check if checkpoint exists, if yes load it, otherwise initialize new agent
    checkpoint_dir = final_logdir / "checkpoints"
    latest_checkpoint = None

    if checkpoint_dir.exists():
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(re.findall(r'\d+', x)[0]))

    if latest_checkpoint:
        print(f"Found existing checkpoint: {latest_checkpoint}. Resuming training...")
        agent = load_checkpoint(checkpoint_dir, latest_checkpoint, log_dir=final_logdir)
    else:
        agent = DreamerAgent(config_dict=config_dict, log_dir=final_logdir)

    agent.train()

if __name__ == "__main__":
    main()