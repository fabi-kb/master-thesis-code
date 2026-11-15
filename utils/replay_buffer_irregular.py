import numpy as np
import torch

class BatchReplayBuffer:

    def __init__(
            self,
            capacity: int,
            time_limit: int, 
            observation_shape: tuple,
            action_shape: tuple,
            batch_size: int
    ):
        
        assert capacity > 0, "Capacity must be a positive integer."
        assert time_limit > 0, "Time limit must be a positive integer."

        self.capacity = capacity
        self.time_limit = time_limit + 1 # An episode of length T has T actions and rewards but T+1 states
        self.position_in_traj = 0
        self.position_in_buffer = 0
        self.size = 0
        self.batch_size = batch_size

        # Pre allocate memory

        self.storage = {
            'action_repeat' : np.zeros((capacity, self.time_limit), dtype=np.float32),
            'observations' : np.zeros((capacity, batch_size, self.time_limit, *observation_shape), dtype=np.float32),
            'actions' : np.zeros((capacity, batch_size, self.time_limit - 1, *action_shape), dtype=np.float32),
            'rewards' : np.zeros((capacity, batch_size, self.time_limit - 1), dtype=np.float32),
            'current_length': np.zeros((capacity,), dtype=np.int32) # to keep track of trajectory lengths
        }


    def add_observation_batch(self, action_repeat, observation, action, reward, first_step=False):
        """
        Add a batch of observations to the replay buffer.
        """

        # If a new episode starts before the current slot is full, close the slot out now
        # instead of stitching two unrelated episodes into one trajectory.
        if first_step and self.position_in_traj != 0:
            self.position_in_buffer = (self.position_in_buffer + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
            self.position_in_traj = 0

        # Validate input shapes
        assert observation.shape[0] == self.batch_size, "Batch size mismatch in observations"
        self.storage['action_repeat'][self.position_in_buffer, self.position_in_traj] = action_repeat
        self.storage['observations'][self.position_in_buffer, :, self.position_in_traj] = observation

        # Add actions and rewards only if not first step
        # This is because for the first step, we don't have a previous action or reward to store
        if not first_step:
            assert action.shape[0] == self.batch_size, "Batch size mismatch in actions"
            assert reward.shape[0] == self.batch_size, "Batch size mismatch in rewards"
            self.storage['actions'][self.position_in_buffer, :, self.position_in_traj - 1] = action
            self.storage['rewards'][self.position_in_buffer, :, self.position_in_traj - 1] = reward


        # Update trajectory length before modulo to ensure correct length
        self.storage['current_length'][self.position_in_buffer] = self.position_in_traj + 1

        
        self.position_in_traj = (self.position_in_traj + 1) % self.time_limit

        if self.position_in_traj == 0: # trajectory ended, move to next buffer position, also circular 
            self.position_in_buffer = (self.position_in_buffer + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        

    def sample_batch(self, sequence_length, device='cpu'):

        # check for valid trajectories with enough sequence length
        valid_indices = np.where(self.storage['current_length'][:self.size + 1] >= sequence_length)[0]
        
        if len(valid_indices) == 0:
            raise ValueError("No trajectories in buffer are long enough yet.")

        # Randomly sample entry/batch from valid indices
        batch_idx = np.random.choice(valid_indices)

        # Start index: sequence_length can start anywhere up to current_length - sequence_length
        max_t = self.storage['current_length'][batch_idx] - sequence_length
        start_idx = np.random.randint(0, max_t + 1)

        batch_data = {}

        # Extract batch data for the sampled trajectory and sequence length + tensor conversion

        batch_data['action_repeat'] = torch.tensor(self.storage['action_repeat'][batch_idx, start_idx : start_idx + sequence_length], dtype=torch.float32).to(device)
        batch_data['observations'] = torch.tensor(self.storage['observations'][batch_idx, :, start_idx : start_idx + sequence_length], dtype=torch.float32).to(device)
        batch_data['actions'] = torch.tensor(self.storage['actions'][batch_idx, :, start_idx : start_idx + sequence_length - 1], dtype=torch.float32).to(device)
        batch_data['rewards'] = torch.tensor(self.storage['rewards'][batch_idx, :, start_idx : start_idx + sequence_length - 1], dtype=torch.float32).to(device)

        return batch_data
     
    