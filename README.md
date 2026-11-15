# Neural SDEs as World Model for Model-Based Reinforcement Learning

Code for the master thesis *Neural SDEs as Worldmodel for Model Based Reinforcement Learning* (Heidelberg University).
A Dreamer-style agent whose world model is either the RSSM or a controlled latent Neural SDE, trained on
irregularly sampled DeepMind Control Suite tasks.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.12. For GPU training, install the CUDA build of PyTorch that matches your system (the experiments used `torch==2.10.0+cu126`).

## Usage

```bash
python agent.py --config config.yaml --logdir runs --opts <key=value> ...
```

`--opts` overrides entries of `config.yaml` in dot notation. Note that `config.yaml` defaults to the RSSM without latent reset and `irregular_max_repeat: 7`; the experiments set these values explicitly:

| Setting | Options |
|---|---|
| Neural SDE (as in the thesis) | `model.name=controlled_nsde model.correction_mode=learnable model.energy_preserving=False` |
| RSSM | `model.name=rssm model.energy_preserving=True` |
| Irregularity level $M$ | `env.irregular_max_repeat=1` (regular), `3`, `5`, `7`, `10` |
| Seed | `env.seed=0` ... `3` |
| Cartpole balance | default |
| Walker stand | `env.name=walker_stand env.dt=0.025` |
| Acrobot swingup | `env.name=acrobot_swingup` |
| Ablation | `env.irregular_max_repeat=1`, `model.correction_mode` in {`none`, `learnable`}, `model.energy_preserving` in {`True`, `False`} |

`energy_preserving=False` resets the latent state every `train.sequence_length` steps.
The thesis reports all results after 400k environment steps; add `train.train_steps=400000` to stop training there.

Example, Neural SDE on walker stand with $M = 5$:

```bash
python agent.py --config config.yaml --logdir runs --opts model.name=controlled_nsde model.correction_mode=learnable model.energy_preserving=False env.name=walker_stand env.dt=0.025 env.irregular_max_repeat=5 env.seed=0
```

## Output

Each run writes to `<logdir>/<options>/`:

- TensorBoard logs (training metrics, `eval/episode_rewards`, `eval/posterior_mse`, imagination metrics)
- `checkpoints/checkpoint_<step>.pth` every 10k environment steps and `final_model.pth`
- `eval_plots/`: posterior reconstructions and prior rollouts of the evaluation episodes

If checkpoints exist in the run directory, training resumes from the latest one. Networks and optimizers are restored, the replay buffer is not.

## Structure

```
agent.py                         training loop, environment interaction, evaluation
config.yaml                      default configuration
model/controlled_nsde.py         controlled latent Neural SDE world model
model/rssm.py                    RSSM world model
model/policy_learning_lambda.py  actor-critic with lambda-returns in imagination
environments/                    DeepMind Control Suite and Gymnasium wrappers
utils/                           replay buffer with irregular sampling, networks, logging, checkpoints, plotting
```

## License

MIT, see [LICENSE](LICENSE).
