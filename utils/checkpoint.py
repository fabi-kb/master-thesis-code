import torch

def save_checkpoint(agent, path, name='checkpoint.pth'):

    checkpoint = {
        'world_model_state_dict': agent.world_model.state_dict(),
        'world_model_optimizer_state_dict': agent.wm_optimizer.state_dict(),
        'policy_state_dict': agent.policy_learning.actor.state_dict(),
        'critic_state_dict': agent.policy_learning.critic.state_dict(),
        'policy_optimizer_state_dict': agent.policy_learning.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.policy_learning.critic_optimizer.state_dict(),
        'config' : agent.config,
        'global_step': agent.global_step
    }

    torch.save(checkpoint, f"{path}/{name}")

    print(f"Checkpoint saved at {path}/{name}")

def load_checkpoint(path, name='checkpoint.pth', log_dir=None):
    from agent import DreamerAgent

    # load checkpoint
    checkpoint = torch.load(f"{path}/{name}", map_location='cpu')

    # initialize agent
    agent = DreamerAgent(config_dict=checkpoint['config'], log_dir=log_dir)

    # load state dicts
    agent.world_model.load_state_dict(checkpoint['world_model_state_dict'])
    agent.wm_optimizer.load_state_dict(checkpoint['world_model_optimizer_state_dict'])
    agent.policy_learning.actor.load_state_dict(checkpoint['policy_state_dict'])
    agent.policy_learning.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.policy_learning.actor_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
    agent.policy_learning.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    agent.global_step = checkpoint.get('global_step', 0)

    print(f"Checkpoint loaded from {path}/{name}")
    return agent


