# Plot_Reward.py — IEEE-aligned reward convergence panel (Sum Rewards + Scientific y-axis)
import os
from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── IEEE SINGLE-COLUMN SETTINGS (match other figures) ──────────
# ── IEEE SINGLE-COLUMN SETTINGS (match other figures) ──────────
IEEE_WIDTH = 3.4   # inches
IEEE_HEIGHT = 2.1  # inches

mpl.rcParams.update({
    "font.family": "serif",
    # STIXGeneral is metrics-compatible with Times New Roman and embeds cleanly
    "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.use14corefonts": False,     # ← important: disable corefonts
    "pdf.fonttype": 42,              # ← embed TrueType fonts
    "ps.fonttype": 42,               # ← embed TrueType for EPS too
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


# ── Helpers ────────────────────────────────────────────────────
def _load_sar_rewards(sar_path: Path) -> np.ndarray:
    """Load SAR (slot-wise) rewards from various formats -> 1D float array."""
    with sar_path.open("rb") as f:
        sar = pickle.load(f)

    records = []
    if isinstance(sar, list):
        for it in sar:
            if isinstance(it, dict) and all(k in it for k in ("ep", "frame", "slot")):
                records.append((it["ep"], it["frame"], it["slot"], it.get("reward", it.get("r"))))
            elif isinstance(it, (list, tuple)) and len(it) >= 4:
                ep, fr, sl, rv = it[0], it[1], it[2], it[3]
                records.append((ep, fr, sl, rv))
    elif isinstance(sar, dict) and "logs" in sar and isinstance(sar["logs"], list):
        for it in sar["logs"]:
            records.append((it["ep"], it["frame"], it["slot"], it.get("reward", it.get("r"))))
    else:
        rewards = None
        for k in ("reward", "rewards", "r"):
            if isinstance(sar, dict) and k in sar:
                rewards = np.asarray(sar[k], dtype=float)
                break
        if rewards is None:
            raise TypeError(f"Unsupported SAR format at {sar_path}")
        return rewards.astype(float)

    if not records:
        raise ValueError(f"No recognizable SAR entries in {sar_path}")

    records.sort(key=lambda t: (t[0], t[1], t[2]))
    vals = [float(rec[3]) for rec in records if rec[3] is not None]
    if len(vals) != len(records):
        raise ValueError(f"{len(records)-len(vals)} entries have reward=None in {sar_path}")
    return np.asarray(vals, dtype=float)


def _episode_reward_series(
    rewards_slotwise: np.ndarray,
    num_slots: int,
    frames_per_episode: int,
    num_episodes: Optional[int] = None,
    mode: str = "sum",  # keep "sum" rewards
) -> np.ndarray:
    """Aggregate slot-wise rewards into episode-wise series."""
    slots_per_episode = int(num_slots) * int(frames_per_episode)
    if slots_per_episode <= 0:
        raise ValueError("slots_per_episode must be positive.")

    total_slots = len(rewards_slotwise)
    n_eps = total_slots // slots_per_episode
    if num_episodes is not None:
        n_eps = min(n_eps, int(num_episodes))
    if n_eps == 0:
        raise ValueError(f"Not enough slots ({total_slots}) for one episode of {slots_per_episode} slots.")

    R = rewards_slotwise[: n_eps * slots_per_episode].reshape(n_eps, slots_per_episode)
    return R.sum(axis=1)


def _moving_avg(y: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return moving average and the x indices it aligns with (start at k-1)."""
    k = int(max(1, k))
    if k == 1 or len(y) < k:
        return y.copy(), np.arange(len(y))
    ma = np.convolve(y, np.ones(k)/k, mode="valid")
    x = np.arange(k-1, k-1+len(ma))
    return ma, x


# ── Main plotting function ─────────────────────────────────────
def plot_reward_convergence_vs_users(
    run_dirs: List[str],
    user_counts: List[int],
    num_slots: int,
    frames_per_episode: int,
    num_episodes: int,
    out_pdf: str = "reward_convergence_vs_users.pdf",
    smooth_k: int = 10,
    faint_alpha: float = 0.25,
    line_alpha: float = 0.95,
    ylabel: str = "Total Reward per Episode",
    title: Optional[str] = None,
    legend_loc: str = "best",
) -> Dict[str, np.ndarray]:
    """
    Load SAR rewards for each run_dir/user_count and plot summed episode-wise reward convergence.
    Keeps scientific y-axis formatting for compact IEEE presentation.
    """
    assert len(run_dirs) == len(user_counts), "run_dirs and user_counts must have same length"

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT), constrained_layout=False)
    episode_index = np.arange(1, num_episodes + 1)
    results: Dict[str, np.ndarray] = {"episode_index": episode_index}

    for run_dir, U in zip(run_dirs, user_counts):
        run_dir = Path(run_dir)

        # find SAR file
        candidates = [
            run_dir / f"sar_logU{U}S{num_slots}.pkl",
            run_dir / "sar_log.pkl",
        ]
        sar_path = next((c for c in candidates if c.exists()), None)
        if sar_path is None:
            picks = sorted(run_dir.glob("*.pkl"))
            sar_path = picks[0] if picks else None
        if sar_path is None or not sar_path.exists():
            raise FileNotFoundError(f"No SAR .pkl found in {run_dir}")

        rewards = _load_sar_rewards(sar_path)
        ep = _episode_reward_series(rewards, num_slots, frames_per_episode, num_episodes, mode="sum")
        results[f"U={U}"] = ep

        # raw faint + smoothed line
        ax.plot(episode_index[:len(ep)], ep, alpha=faint_alpha, linewidth=0.9, label=None)
        ma, mx = _moving_avg(ep, smooth_k)
        ax.plot(episode_index[mx], ma, alpha=line_alpha, linewidth=1.2, label=fr"$M$={U}")

    # axis labels + formatting
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.4)
    ax.legend(frameon=False, loc=legend_loc, ncol=1)
    ax.set_xlim(1, num_episodes)
    ax.margins(x=0)

    # scientific notation for large rewards
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useMathText=True)

    # adjust margins
    fig.subplots_adjust(left=0.18, right=0.95, bottom=0.20, top=0.9)

    # save
    out_path = Path(out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.as_posix(), format="pdf", dpi=600)  # fixed-size canvas, no bbox_inches
    plt.close(fig)

    return results


# ── Example usage ──────────────────────────────────────────────
if __name__ == "__main__":
    num_slots = 15
    frames_per_episode = 200
    num_episodes = 100
    user_counts = [60, 65, 70, 75]
    Base_PPO = "AoI_PPO_Users"
    run_dirs = [os.path.join(Base_PPO, f"AoI_U{u}_S{num_slots}_EP{num_episodes}_RewardNS") for u in user_counts]

    plot_reward_convergence_vs_users(
        run_dirs,
        user_counts,
        num_slots,
        frames_per_episode,
        num_episodes,
        out_pdf="reward_convergence_vs_users.png",
        smooth_k=6,
        faint_alpha=0.25,
        line_alpha=0.95,
        ylabel="Total Reward per Episode",
        title=None,
        legend_loc="best",
    )
