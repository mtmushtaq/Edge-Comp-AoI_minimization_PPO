import os
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# --- existing helper (already defined in your codebase) ---
# def episode_end_avg_series(conn, num_slots, frames_per_episode, num_episodes): ...
# def mean_of_episode_end_averages_for_run(...):  # as given in your message
def episode_end_avg_series(conn: sqlite3.Connection,
                           num_slots: int,
                           frames_per_episode: int,
                           num_episodes: int) -> np.ndarray:
    series = np.full(num_episodes, np.nan, dtype=float)
    cur = conn.cursor()
    for e in range(1, num_episodes + 1):
        cum_sum, n_seen, last_m = {}, {}, {}
        q = ("SELECT uid, aoi FROM logs "
             "WHERE ep=? ORDER BY frame ASC, slot ASC, uid ASC")
        for uid, aoi in cur.execute(q, (e,)):
            uid = int(uid); aoi = float(aoi)
            if uid not in n_seen:
                n_seen[uid] = 0
                cum_sum[uid] = 0.0
            n_seen[uid] += 1
            cum_sum[uid] += aoi
            last_m[uid] = cum_sum[uid] / n_seen[uid]
        if last_m:
            series[e - 1] = float(np.mean(list(last_m.values())))
    return series


def mean_of_episode_end_averages_for_run(run_dir: str,
                                         num_slots: int,
                                         frames_per_episode: int,
                                         num_episodes: int,
                                         db_name: str = "slotwise_data.sqlite") -> Tuple[float, np.ndarray]:
    db_path = os.path.join(run_dir, db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        ser = episode_end_avg_series(conn, num_slots, frames_per_episode, num_episodes)
    finally:
        conn.close()
    return float(np.nanmean(ser)), ser


import os
import sqlite3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Tuple

# ─── IEEE SINGLE-COLUMN SETTINGS ───────────────────────────────
IEEE_WIDTH = 3.4
IEEE_HEIGHT = 2.1
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
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})



# ─── MULTI-POLICY COMPARISON PLOT ──────────────────────────────
def plot_mean_episode_end_avg_vs_users_multi(
    run_dirs_PPO: List[str],
    run_dirs_RNDM: List[str],
    run_dirs_Greedy: List[str],
    user_counts: List[int],
    num_slots: int,
    frames_per_episode: int,
    num_episodes: int,
    out_pdf: str = "mean_episode_end_avg_vs_users_multi.pdf"
):
    """
    Plots mean episode-end AoI vs number of users for three scheduling policies:
    Proposed PPO, Random, and Greedy.
    Uses IEEE single-column figure sizing and style.
    """

    # sanity
    if not (len(run_dirs_PPO) == len(run_dirs_RNDM) == len(run_dirs_Greedy) == len(user_counts)):
        raise ValueError("All run_dir lists and user_counts must have the same length")

    # helper to compute mean AoI for each run directory
    def compute_policy_means(run_dirs, policy_name):
        vals = []
        for M, rd in zip(user_counts, run_dirs):
            try:
                mean_val, _ = mean_of_episode_end_averages_for_run(rd, num_slots, frames_per_episode, num_episodes)
                print(f"[{policy_name}] M={M:<3d} → Mean AoI={mean_val:.3f}")
            except Exception as e:
                print(f"[WARN] {policy_name} | M={M}: {e}")
                mean_val = np.nan
            vals.append(mean_val)
        return np.array(vals, dtype=float)

    y_ppo    = compute_policy_means(run_dirs_PPO,    "Proposed PPO")
    y_random = compute_policy_means(run_dirs_RNDM,   "Random")
    y_greedy = compute_policy_means(run_dirs_Greedy, "Greedy")

    # ─── Plot setup ─────────────────────────────────────────────
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.xticks(user_counts, [str(m) for m in user_counts])

    plt.plot(user_counts, y_ppo, color="#6A0DAD", marker="o",
             markerfacecolor="none", markeredgecolor="#6A0DAD",
             markersize=4.8, linewidth=1.4, label="Proposed PPO")

    plt.plot(user_counts, y_random, color="#555555", marker="s",
             markerfacecolor="none", markeredgecolor="#555555",
             markersize=4.8, linewidth=1.3, label="Random")

    plt.plot(user_counts, y_greedy, color="#2E8B57", marker="^",
             markerfacecolor="none", markeredgecolor="#2E8B57",
             markersize=4.8, linewidth=1.3, label="Greedy")

    xmin, xmax = min(user_counts), max(user_counts)
    pad = max(1, int(0.03 * (xmax - xmin if xmax > xmin else 2)))
    plt.xlim(xmin - pad, xmax + pad)
    plt.xlabel(r"Number of Users ($M$)")
    plt.ylabel("Average AoI ")
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend(frameon=False, loc="best", ncol=1)
    plt.tight_layout(pad=0.3)
    plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[PLOT] Saved → {out_pdf}")
    return {
        "user_counts": user_counts,
        "Proposed PPO": y_ppo,
        "Random": y_random,
        "Greedy": y_greedy,
    }

user_counts = [60, 65, 70, 75]
num_slots = 15
frames_per_episode = 200
num_episodes = 100

run_dirs_PPO    = [f"AoI_U{u}_S{num_slots}_EP{num_episodes}_RewardNS" for u in user_counts]
run_dirs_RNDM   = [f"AoI_U{u}_S{num_slots}_EP{num_episodes}_RNDM"  for u in user_counts]
run_dirs_Greedy = [f"AoI_U{u}_S{num_slots}_EP{num_episodes}_GREEDY" for u in user_counts]
run_dirs_Threshold = [f"AoI_U{u}_S{num_slots}_EP{num_episodes}_TH" for u in user_counts]

plot_mean_episode_end_avg_vs_users_multi(
    run_dirs_PPO, run_dirs_RNDM, run_dirs_Greedy,
    user_counts, num_slots, frames_per_episode, num_episodes,
    out_pdf="AoI_vs_Users_PPO_RNDM_Greedy_IEEE.pdf"
)
