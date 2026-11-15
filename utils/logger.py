import torch
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class Logger:

    def __init__(self, log_dir: str):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def log_scalar(self, tag, value, step):


        self.writer.add_scalar(tag, value, step)

    
    def log_training_metrics(self, metrics: dict, step: int):

        for key, value in metrics.items():
            self.log_scalar(f"train/{key}", value, step)

    def log_imagination_metrics(self, metrics: dict, step: int):
        
        for key, value in metrics.items():
            self.log_scalar(f"imagination/{key}", value, step)


    def log_evaluation_metrics(self, metrics: dict, step: int):

        for key, value in metrics.items():
            value = np.mean(value) if isinstance(value, list) else value  # take mean if it's a list of values
            self.log_scalar(f"eval/{key}", value, step)




    def close(self):
        self.writer.close()