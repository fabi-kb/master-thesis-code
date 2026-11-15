

class Environment:
    
    @property
    def observation_space(self):
        raise NotImplementedError("Subclasses must implement the observation_space property.")
    
    @property
    def action_space(self):
        raise NotImplementedError("Subclasses must implement the action_space property.")
    
    @property
    def metadata(self):
        return {}
    
    @property
    def render_mode(self):
        return None
    

    def step(self, action):
        raise NotImplementedError("Subclasses must implement the step method.")
    
    def reset(self, seed=None, options=None):
        raise NotImplementedError("Subclasses must implement the reset method.")
    
    def close(self):
        pass

    