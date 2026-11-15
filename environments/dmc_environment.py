import gymnasium as gym

import numpy as np
from environments.base_env import Environment


class DMCEnvironment(Environment):

    def __init__(
        self,
        name: str, 
        time_limit: int = None,
        dt: float = None,
        seed: int = None,
        action_repeat: int = 1,
    ):
        domain, task = name.split("_", 1)
        if domain == "cup": # only domain with multiple words
            domain = "ball_in_cup"

        from dm_control import suite 

        env_kwargs = {}
        if dt is not None:
            env_kwargs["control_timestep"] = dt

        task_kwargs = {}
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit * dt * action_repeat if dt is not None else time_limit
        if seed is not None:
            task_kwargs["random"] = seed

        try:
            self._env = suite.load(
                domain_name=domain,
                task_name=task,
                environment_kwargs=env_kwargs,
                task_kwargs=task_kwargs,
            )
        except TypeError as e:
            if "control_timestep" not in str(e) or "control_timestep" not in env_kwargs:
                raise
            # This domain (e.g. walker) hardcodes its own control_timestep and
            # doesn't accept an override — fall back to its native value.
            fallback_kwargs = {k: v for k, v in env_kwargs.items() if k != "control_timestep"}
            self._env = suite.load(
                domain_name=domain,
                task_name=task,
                environment_kwargs=fallback_kwargs,
                task_kwargs=task_kwargs,
            )

        if dt is not None:
            actual_dt = self._env._n_sub_steps * self._env.physics.timestep()
            if not np.isclose(actual_dt, dt):
                raise ValueError(
                    f"{name}: this domain hardcodes control_timestep={actual_dt}, "
                    f"but config dt={dt} was requested. Set env.dt={actual_dt} in "
                    f"the config for this task — time_limit and irregular-sampling "
                    f"both assume dt matches the environment's actual control_timestep."
                )

        self._time_limit = time_limit
        self.steps_taken = 0

        self.action_repeat = action_repeat


        # image observations are not supported


    @property
    def observation_space(self) -> gym.spaces.Box:
        observation_dim = 0

        for key in sorted(self._env.observation_spec().keys()):
            spec = self._env.observation_spec()[key]
            shape = tuple(spec.shape)
            observation_dim += int(np.prod(shape)) if len(shape) > 0 else 1

        low = -np.inf * np.ones((observation_dim,), dtype=np.float32)
        high = np.inf * np.ones((observation_dim,), dtype=np.float32)

        return gym.spaces.Box(low=low, high=high, shape=(observation_dim,), dtype=np.float32)

    @property
    def action_space(self) -> gym.spaces.Box:
        spec = self._env.action_spec()
        return gym.spaces.Box(low=spec.minimum, high=spec.maximum, shape=spec.shape, dtype=np.float32)
        

    def step(self, action):

        terminated = False
        limit_reached = False
        reward = 0.0
        self.steps_taken += 1

        for _ in range(self.action_repeat):

            timestep = self._env.step(action)
            reward += timestep.reward or 0.0

            terminated = bool(timestep.last())
            if terminated:
                break

        limit_reached = self._time_limit is not None and self.steps_taken >= self._time_limit

        obs = self._flatten_observation(timestep.observation)

        return obs, reward, terminated, limit_reached,{}



    def reset(self, seed=None, options=None):
        timestep = self._env.reset()
        obs = self._flatten_observation(timestep.observation)
        self.steps_taken = 0
        return obs, {}

    def close(self):
        pass


    # Helper functions
    def _flatten_observation(self, obs):
        parts = []

        for key in sorted(obs.keys()): 
            val = np.asarray(obs[key], dtype=np.float32).ravel()
            parts.append(val)
        if not parts:
            return np.zeros((0,), dtype=np.float32)
        
        return np.concatenate(parts, axis=0)
