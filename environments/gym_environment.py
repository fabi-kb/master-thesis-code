import gymnasium as gym
import numpy as np

from environments.base_env import Environment

class GymEnvironment(Environment):

    def __init__(
        self, 
        name: str, 
        time_limit: int = None,
        dt: float = None, 
        seed: int = None,
        action_repeat: int = 1,
    ):
        
        if time_limit is not None:
            self._env = gym.make(name, max_episode_steps=time_limit * action_repeat)
        else:
            self._env = gym.make(name)
             
        self._dt = dt
        self._seed = seed
        self.action_repeat = action_repeat

        if dt is not None:
            if hasattr(self._env.unwrapped, "dt"):
                self._env.unwrapped.dt = dt
            else:
                print(f"Warning: Environment {name} does not have a dt attribute. Ignoring dt parameter.")

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._env.observation_space
    
    @property
    def action_space(self) -> gym.spaces.Box:
        return self._env.action_space
    
    @property
    def metadata(self) -> dict:
        return self._env.metadata
    
    @property
    def render_mode(self) -> str:
        return self._env.render_mode

    def step(self, action): 

        terminated_flag = False # natural end of episode
        limit_reached = False # episode ended due to time limit or other constraints
        info = {}
        reward = 0.0

        for _ in range(self.action_repeat): 

            obs, reward_intermediate, terminated, limit_reached, info = self._env.step(action)

            terminated_flag = bool(terminated_flag or terminated)

            reward += reward_intermediate or 0.0 

            if terminated_flag or limit_reached:
                break

        return obs, reward, terminated_flag, limit_reached, info
    
    def reset(self, seed=None, options=None):
        self.steps_taken = 0
        observation, _ = self._env.reset(seed=seed, options=options)
        return observation, {}
    
    def close(self):
        return self._env.close()



            


