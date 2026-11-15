import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


_ROLLOUT_COLORS = ['#e74c3c', '#2980b9', '#e67e22']  # red, blue, orange — max 3
_PRIOR_1STEP_COLOR = '#8e44ad'                         # purple for per-step prior


def _select_rollouts(prior_rollouts, n=3):
    """Pick n evenly-spaced rollouts from the collected list."""
    if len(prior_rollouts) <= n:
        return prior_rollouts
    indices = [round(i * (len(prior_rollouts) - 1) / (n - 1)) for i in range(n)]
    return [prior_rollouts[i] for i in indices]


def plot_evaluation(
    episode_idx,
    global_step,
    observations,
    posterior_recons,
    prior_rollouts,
    prior_1step,
    save_dir,
    dt=0.05,
    dim_names=None,
    max_prior_shown=3,
    reset_times=None,
):
    """
    Save two figures per episode:

      *_posterior.png  — ground truth vs posterior reconstruction (per step)
      *_prior.png      — ground truth vs 1-step prior prediction (per step)
                         plus max_prior_shown multi-step imagination rollouts

    Args:
        observations:     np.ndarray (T, obs_dim)
        posterior_recons: np.ndarray (T, obs_dim), aligned with observations
        prior_rollouts:   list of {'start': int, 'decoded': np.ndarray (horizon, obs_dim)}
        prior_1step:      np.ndarray (T, obs_dim) — per-step 1-step-ahead prior prediction
        save_dir:         str or Path
        dt:               seconds per step
        dim_names:        optional list[str] of length obs_dim
        max_prior_shown:  how many multi-step rollouts to draw (evenly sampled)

    Returns:
        (posterior_path, prior_path)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    obs = np.array(observations)
    post = np.array(posterior_recons)
    p1s = np.array(prior_1step)

    T, obs_dim = obs.shape
    time_axis = np.arange(T) * dt

    if dim_names is None:
        dim_names = [f'Dim {i}' for i in range(obs_dim)]

    stem = f'ep{episode_idx:04d}_step{global_step:07d}'

    resets = reset_times or []

    posterior_path = _plot_posterior(
        obs, post, time_axis, obs_dim, dim_names,
        save_dir / f'{stem}_posterior.png',
        episode_idx, global_step, resets,
    )

    selected = _select_rollouts(prior_rollouts, n=max_prior_shown)
    prior_path = _plot_prior(
        obs, p1s, selected, time_axis, obs_dim, dim_names,
        save_dir / f'{stem}_prior.png',
        episode_idx, global_step, dt, resets,
    )

    return posterior_path, prior_path


# ---------------------------------------------------------------------------

def _make_fig(obs_dim, title):
    fig, axes = plt.subplots(
        obs_dim, 1,
        figsize=(14, 3.4 * obs_dim),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.005)
    return fig, axes[:, 0]


def _finish_axes(ax, dim_name, is_last):
    ax.set_ylabel(dim_name, fontsize=9, labelpad=4)
    ax.grid(True, alpha=0.2, linewidth=0.7)
    ax.tick_params(labelsize=8)
    if is_last:
        ax.set_xlabel('Time (s)', fontsize=9)


def _draw_resets(axes, reset_times):
    """Draw a vertical dashed grey line at every latent-state reset."""
    for t in reset_times:
        for ax in axes:
            ax.axvline(x=t, color='#7f8c8d', linewidth=1.2,
                       linestyle='--', alpha=0.7, zorder=6)


def _plot_posterior(obs, post, time_axis, obs_dim, dim_names, out_path,
                    episode_idx, global_step, reset_times):
    fig, axes = _make_fig(
        obs_dim,
        f'Posterior Reconstruction  —  Episode {episode_idx}  |  Step {global_step}',
    )
    for d, ax in enumerate(axes):
        ax.plot(time_axis, obs[:, d], color='#2c3e50', linewidth=1.8, label='Ground truth')
        ax.plot(time_axis, post[:, d], color='#27ae60', linewidth=1.5,
                linestyle='--', label='Posterior')
        _finish_axes(ax, dim_names[d], d == obs_dim - 1)

    _draw_resets(axes, reset_times)

    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    if reset_times:
        legend_handles.append(
            plt.Line2D([0], [0], color='#7f8c8d', linewidth=1.2,
                       linestyle='--', label='Latent reset')
        )
        legend_labels.append('Latent reset')
    axes[0].legend(legend_handles, legend_labels, loc='upper right',
                   fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out_path


def _plot_prior(obs, prior_1step, selected_rollouts, time_axis, obs_dim, dim_names,
                out_path, episode_idx, global_step, dt, reset_times):
    fig, axes = _make_fig(
        obs_dim,
        f'Prior Imagination  —  Episode {episode_idx}  |  Step {global_step}',
    )

    legend_handles = [
        plt.Line2D([0], [0], color='#2c3e50', linewidth=1.8, label='Ground truth'),
        plt.Line2D([0], [0], color=_PRIOR_1STEP_COLOR, linewidth=1.4,
                   linestyle='--', label='Prior 1-step ahead'),
    ]
    for r_idx, rollout in enumerate(selected_rollouts):
        color = _ROLLOUT_COLORS[r_idx % len(_ROLLOUT_COLORS)]
        legend_handles.append(
            plt.Line2D([0], [0], color=color, linewidth=1.5, linestyle=':',
                       marker='o', markersize=4,
                       label=f'Prior rollout @ t={rollout["start"] * dt:.1f} s')
        )
    if reset_times:
        legend_handles.append(
            plt.Line2D([0], [0], color='#7f8c8d', linewidth=1.2,
                       linestyle='--', label='Latent reset')
        )

    for d, ax in enumerate(axes):
        ax.plot(time_axis, obs[:, d], color='#2c3e50', linewidth=1.8, zorder=3)
        ax.plot(time_axis, prior_1step[:, d], color=_PRIOR_1STEP_COLOR,
                linewidth=1.4, linestyle='--', zorder=4)

        for r_idx, rollout in enumerate(selected_rollouts):
            start = rollout['start']
            decoded = rollout['decoded']
            horizon = decoded.shape[0]
            roll_time = np.arange(start, start + horizon) * dt
            color = _ROLLOUT_COLORS[r_idx % len(_ROLLOUT_COLORS)]

            ax.axvspan(roll_time[0], roll_time[-1], alpha=0.07, color=color, zorder=1)
            ax.axvline(x=roll_time[0], color=color, linewidth=1.0,
                       linestyle='--', alpha=0.55, zorder=2)
            marker_every = max(1, horizon // 20)
            ax.plot(roll_time, decoded[:, d], color=color, linewidth=1.5,
                    linestyle=':', marker='o', markersize=3,
                    markevery=marker_every, zorder=5)

        _finish_axes(ax, dim_names[d], d == obs_dim - 1)

    _draw_resets(axes, reset_times)

    axes[0].legend(handles=legend_handles, loc='upper right',
                   fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out_path
