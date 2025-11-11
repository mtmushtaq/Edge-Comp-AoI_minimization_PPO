# Write a reusable plotting utility that creates IEEE-friendly, episode-wise reward
# convergence plots across multiple runs (e.g., different user_counts). It keeps
# some jitter by using a small moving average window and plotting raw curves faintly.
# The function will look for sar logs in each run dir and generate a single PDF.
#
# Saved as /mnt/data/Plot_Reward.py

from pathlib import Path
import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

#from networkx.algorithms.bipartite.cluster import modes

# --- IEEE figure defaults (single-column) ---
IEEE_WIDTH = 3.4   # inches (per IEEE column)
IEEE_HEIGHT = 2.1  # inches
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.use14corefonts": True,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.8,
    "axes.titlesize": 9,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.1,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})


def _load_sar_rewards(sar_path: Path) -> np.ndarray:
    """
    Load SAR (slot-wise) rewards from a variety of expected formats and
    return a 1D float numpy array 'rewards'.
    """
    with sar_path.open("rb") as f:
        sar = pickle.load(f)

    records = []
    if isinstance(sar, list):
        for it in sar:
            if isinstance(it, dict) and all(k in it for k in ("ep", "frame", "slot")):
                records.append((it["ep"], it["frame"], it["slot"],
                                it.get("reward", it.get("r"))))
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

    # Sort and extract rewards
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
    mode: str = "avg_per_slot",
) -> np.ndarray:
    """
    Convert slot-wise rewards to an episode-wise series.
    mode: "avg_per_slot" (default) or "sum".
    """
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
    if mode == "sum":
        ep = R.sum(axis=1)
    else:  # "avg_per_slot"
        ep = R.mean(axis=1)
    return ep


def _moving_avg(y: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return moving average and the x indices it aligns with (start at k-1)."""
    k = int(max(1, k))
    if k == 1 or len(y) < k:
        return y.copy(), np.arange(len(y))
    ma = np.convolve(y, np.ones(k)/k, mode="valid")
    x = np.arange(k-1, k-1+len(ma))
    return ma, x


def plot_reward_convergence_vs_users(
    run_dirs: List[str],
    user_counts: List[int],
    num_slots: int,
    frames_per_episode: int,
    num_episodes: int,
    out_pdf: str = "reward_convergence_vs_users.pdf",
    smooth_k: int = 5,            # small window to keep some jitter
    faint_alpha: float = 0.25,    # raw curve transparency
    line_alpha: float = 0.95,     # smoothed curve visibility
    ylabel: str = "Average Reward per Episode (per slot)",
    title: Optional[str] = None,
    legend_loc: str = "best",
) -> Dict[str, np.ndarray]:
    """
    For each run_dir/user_count, load SAR rewards and plot episode-wise reward series
    to show convergence. Keeps some jitter by using a small moving-average window.
    Saves a single IEEE-friendly PDF.
    Returns a dict with episode index and per-run series for further analysis.
    """
    assert len(run_dirs) == len(user_counts), "run_dirs and user_counts must have same length"

    # Prepare figure
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))

    episode_index = np.arange(1, num_episodes + 1)
    results: Dict[str, np.ndarray] = {"episode_index": episode_index}

    for run_dir, U in zip(run_dirs, user_counts):
        run_dir = Path(run_dir)
        # Try common SAR filename patterns
        candidates = [
            run_dir / f"sar_logU{U}S{num_slots}.pkl",
            run_dir / "sar_log.pkl",
        ]
        sar_path = None
        for c in candidates:
            if c.exists():
                sar_path = c
                break
        if sar_path is None:
            # Fallback: first *.pkl in run_dir
            picks = sorted(run_dir.glob("*.pkl"))
            if picks:
                sar_path = picks[0]
        if sar_path is None or not sar_path.exists():
            raise FileNotFoundError(f"No SAR .pkl found in {run_dir}")

        rewards = _load_sar_rewards(sar_path)
        ep = _episode_reward_series(rewards, num_slots, frames_per_episode, num_episodes, mode="sum")

        # store
        results[f"U={U}"] = ep

        # raw (faint) + lightly smoothed line
        plt.plot(episode_index[:len(ep)], ep, alpha=faint_alpha, linewidth=0.9, label=None)
        ma, mx = _moving_avg(ep, smooth_k)
        lbl = fr"$M$={U}"
        plt.plot(episode_index[mx], ma, alpha= line_alpha, linewidth=1.2, label=lbl)

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.35, linestyle="--", linewidth=0.4)
    plt.legend(frameon=False, loc=legend_loc, ncol=1)
    plt.tight_layout(pad=0.3)

    # Ensure output directory exists
    out_path = Path(out_pdf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path.as_posix(), format="pdf", dpi=600, bbox_inches="tight")
    plt.close()

    return results



num_slots = 15
frames_per_episode = 200
num_episodes = 100
user_counts = [60, 65, 70, 75]
run_dirs = [f"AoI_U{u}_S{num_slots}_EP{num_episodes}_RewardNS" for u in user_counts]

plot_reward_convergence_vs_users(
    run_dirs, user_counts, num_slots, frames_per_episode, num_episodes,
    out_pdf="reward_convergence_vs_users.pdf",   # output name
    smooth_k=10,                                   # small MA to keep jitters
    faint_alpha=0.25, line_alpha=0.95,            # faint raw, clear MA
    ylabel="Average Reward per Episode2",
    title=None,                                   # or e.g. "Reward Convergence"
    legend_loc="best"
)
