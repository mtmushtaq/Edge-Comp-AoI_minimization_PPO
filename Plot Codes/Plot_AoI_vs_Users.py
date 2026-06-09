import os
import sqlite3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Tuple

# IEEE single-column sizing
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

def plot_mean_episode_end_avg_vs_users(run_dirs: List[str],
                                       user_counts: List[int],
                                       num_slots: int,
                                       frames_per_episode: int,
                                       num_episodes: int,
                                       out_pdf: str = "mean_episode_end_avg_vs_users.pdf",
                                       label: str = "Proposed PPO",
                                       line_color: str = "#6A0DAD",   # purple line
                                       marker_color: str = "green",   # green markers
                                       marker: str = "o",
                                       show_line: bool = True) -> Tuple[List[int], List[float]]:
    if len(run_dirs) != len(user_counts):
        raise ValueError("run_dirs and user_counts must have the same length")
    y_vals: List[float] = []
    for M, run_dir in zip(user_counts, run_dirs):
        try:
            y, _ = mean_of_episode_end_averages_for_run(run_dir, num_slots, frames_per_episode, num_episodes)
        except Exception:
            y = float("nan")
        y_vals.append(y)
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.xticks(user_counts, [str(m) for m in user_counts])
    if show_line:
        plt.plot(user_counts, y_vals,
                 color=line_color,
                 marker=marker,
                 markerfacecolor="none",
                 markeredgecolor=marker_color,
                 markersize=5,
                 linewidth=1.6,
                 label=label)
    else:
        plt.scatter(user_counts, y_vals,
                    color=marker_color,
                    marker=marker,
                    s=18,
                    label=label)
    xmin, xmax = min(user_counts), max(user_counts)
    pad = max(1, int(0.03 * (xmax - xmin if xmax > xmin else 2)))
    plt.xlim(xmin - pad, xmax + pad)
    plt.xlabel(r"Number of Users ($M$)")
    plt.ylabel("Network Average AoI")
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout(pad=0.3)
    plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close()
    return user_counts, y_vals

if __name__ == "__main__":
    num_slots = 15
    frames_per_episode = 200
    num_episodes = 100
    user_counts = [60, 65, 70, 75]
    run_dirs = [f"AoI_U{u}_S{num_slots}_EP{num_episodes}_Greedy" for u in user_counts]

    plot_mean_episode_end_avg_vs_users(run_dirs, user_counts,
                                       num_slots, frames_per_episode, num_episodes,
                                       out_pdf="min_aoi_vs_users_greedy.pdf",
                                       label="Greedy",
                                       line_color="#6A0DAD",
                                       marker_color="purple",
                                       marker="o",
                                       show_line=True)
