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

# ---------- DB helpers ----------
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

def episode_true_avg(conn, num_slots, frames_per_episode, num_episodes, expect_M=None):
    series = np.full(num_episodes, np.nan, float)
    cur = conn.cursor()
    slots_per_ep = num_slots * frames_per_episode
    for e in range(1, num_episodes + 1):
        # count unique users present (optional, to check completeness)
        if expect_M is not None:
            M = cur.execute("SELECT COUNT(DISTINCT uid) FROM logs WHERE ep=?", (e,)).fetchone()[0] or 0
            if M and expect_M and M != expect_M:
                print(f"[WARN] ep={e}: DISTINCT uid={M} != expected M={expect_M}")
        # average across all rows in the episode
        row = cur.execute("SELECT AVG(aoi), COUNT(*) FROM logs WHERE ep=?", (e,)).fetchone()
        if row and row[1] > 0:
            series[e-1] = float(row[0])
    return series


def episode_end_avg_series_TH(conn: sqlite3.Connection,
                           num_slots: int,
                           frames_per_episode: int,
                           num_episodes_TH: int) -> np.ndarray:
    series = np.full(num_episodes, np.nan, dtype=float)
    cur = conn.cursor()
    for e in range(1, num_episodes_TH + 1):
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
        ser = episode_true_avg(conn, num_slots, frames_per_episode, num_episodes)
    finally:
        conn.close()
    return float(np.nanmean(ser)), ser

def mean_of_episode_end_averages_for_run_TH(run_dir: str,
                                         num_slots: int,
                                         frames_per_episode: int,
                                         num_episodes_TH: int,
                                         db_name: str = "slotwise_data.sqlite") -> Tuple[float, np.ndarray]:
    db_path = os.path.join(run_dir, db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        ser = episode_end_avg_series_TH(conn, num_slots, frames_per_episode, num_episodes_TH)
    finally:
        conn.close()
    return float(np.nanmean(ser)), ser

# ---------- Directory builders (exact patterns you gave) ----------
import os

def build_dir_proposed(u: int, num_slots: int, num_episodes: int, gamma_th_db: int) -> str:
    """
    Proposed PPO (GTH) directory structure:
      Base: AoI_U{u}_S{S}_GTH
      Sub:  AoI_U{u}_S{S}_EP{EP}_GTH{±x}
    Example:
      AoI_U60_S15_GTH/AoI_U60_S15_EP50_GTHminus10
    """
    base = f"AoI_U{u}_S{num_slots}_GTH"
    sub = f"AoI_U{u}_S{num_slots}_EP{num_episodes}_GTH{('minus' + str(abs(gamma_th_db))) if gamma_th_db < 0 else gamma_th_db}"
    return os.path.join(base, sub)


def build_dir_threshold_v2(u: int, num_slots: int, num_episodes: int, gamma_th_db: int) -> str:
    """
    Threshold Policy V2 (Policy) directory structure:
      Base: AoI_U{u}_S{S}_Policy
      Sub:  AoI_U{u}_S{S}_EP{EP}TH{±x}_PolicyV2
    Example:
      AoI_U60_S15_Policy/AoI_U60_S15_EP50THminus10_PolicyV2
    """
    base = f"Threshold_User_{u}"
    sub = f"AoI_U{u}_S{num_slots}_EP{num_episodes}TH{('minus' + str(abs(gamma_th_db))) if gamma_th_db < 0 else gamma_th_db}_PolicyV2"
    return os.path.join(base, sub)


def build_dir_greedy(u: int, num_slots: int, num_episodes: int, gamma_th_db: int) -> str:
    """
    Greedy policy directory structure:
      Base: AoI_U{u}_S{S}_Greedy
      Sub:  AoI_U{u}_S{S}_EP{EP}_Greedy_TH{±x}
    Example:
      AoI_U60_S15_Greedy/AoI_U60_S15_EP50_Greedy_THminus10
    """
    base = f"AoI_U{u}_S{num_slots}_Greedy"
    sub = f"AoI_U{u}_S{num_slots}_EP{num_episodes}_Greedy_TH{('minus' + str(abs(gamma_th_db))) if gamma_th_db < 0 else gamma_th_db}"
    return os.path.join(base, sub)


def build_dir_random(u: int, num_slots: int, num_episodes: int, gamma_th_db: int) -> str:
    """
    Random policy directory structure:
      Base: AoI_U{u}_S{S}_THRNDM
      Sub:  AoI_U{u}_S{S}_EP{EP}_TH{±x}RNDM
    Example:
      AoI_U60_S15_THRNDM/AoI_U60_S15_EP50_THminus10RNDM
    """
    base = f"AoI_U{u}_S{num_slots}_THRNDM"
    sub = f"AoI_U{u}_S{num_slots}_EP{num_episodes}_TH{('minus' + str(abs(gamma_th_db))) if gamma_th_db < 0 else gamma_th_db}RNDM"
    return os.path.join(base, sub)


# ---------- Plotter: FOUR policies on one figure ----------
def plot_aoi_vs_sinr_threshold_four_policies(
    thresholds_db: List[int],
    num_users: int,
    num_slots: int,
    frames_per_episode: int,
    num_episodes: int,
    num_episodes_TH: int,
    base_dir: str = ".",
    db_name: str = "slotwise_data.sqlite",
    out_pdf: str = "aoi_vs_sinr_threshold_four_policies.pdf",
) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    x_vals = list(thresholds_db)

    def collect(builder):
        if builder == build_dir_threshold_v2:
            vals = []
            for th in x_vals:
                run_dir = builder(num_users, num_slots, num_episodes, int(th))
                run_path = os.path.join(base_dir, run_dir)
                try:
                    y, _ = mean_of_episode_end_averages_for_run_TH(run_path, num_slots, frames_per_episode, num_episodes_TH,
                                                                db_name=db_name)
                except Exception:
                    y = float("nan")
                vals.append(y)
            return vals
        else:
            vals = []
            for th in x_vals:
                run_dir = builder(num_users, num_slots, num_episodes, int(th))
                run_path = os.path.join(base_dir, run_dir)
                try:
                    y, _ = mean_of_episode_end_averages_for_run(run_path, num_slots, frames_per_episode, num_episodes, db_name=db_name)
                except Exception:
                    y = float("nan")
                vals.append(y)
            return vals

    y_ppo      = collect(build_dir_proposed)
    y_threshold= collect(build_dir_threshold_v2)
    y_greedy   = collect(build_dir_greedy)
    y_random   = collect(build_dir_random)

    # --- Plot with your palette ---
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.xticks(x_vals, [f"{int(v)}" for v in x_vals])

    # PPO
    plt.plot(x_vals, y_ppo, color="#6A0DAD", marker="o",
             markerfacecolor="none", markeredgecolor="#6A0DAD",
             markersize=4.8, linewidth=1.4, label="Proposed PPO")

    # Random
    plt.plot(x_vals, y_random, color="#555555", marker="s",
             markerfacecolor="none", markeredgecolor="#555555",
             markersize=4.8, linewidth=1.3, label="Random")

    # Greedy
    plt.plot(x_vals, y_greedy, color="#2E8B57", marker="^",
             markerfacecolor="none", markeredgecolor="#2E8B57",
             markersize=4.8, linewidth=1.3, label="Greedy")

    # Threshold
    plt.plot(x_vals, y_threshold, color="#D35400", marker="D",
             markerfacecolor="none", markeredgecolor="#D35400",
             markersize=4.8, linewidth=1.3, label="Threshold")

    xmin, xmax = min(x_vals), max(x_vals)
    span = xmax - xmin if xmax > xmin else 2
    pad = max(1, int(0.03 * span))
    plt.xlim(xmin - pad, xmax + pad)

    plt.xlabel(r"SINR Threshold $\gamma_{\mathrm{th}}$ (dB)")
    plt.ylabel("Average AoI")
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout(pad=0.3)
    plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close()

    return x_vals, y_ppo, y_random, y_greedy, y_threshold

# ---------- Main ----------
if __name__ == "__main__":
    num_users = 60
    num_slots = 15
    frames_per_episode = 200
    num_episodes = 50
    num_episodes_TH = 100

    thresholds_db = [-20, -15, -10, -5, 0]

    plot_aoi_vs_sinr_threshold_four_policies(
        thresholds_db=thresholds_db,
        num_users=num_users,
        num_slots=num_slots,
        frames_per_episode=frames_per_episode,
        num_episodes=num_episodes,
        num_episodes_TH = num_episodes_TH,
        base_dir=".",                       # change if runs are inside a folder
        db_name="slotwise_data.sqlite",
        out_pdf="aoi_vs_sinr_threshold_3_four_policies_mean.pdf",
    )
