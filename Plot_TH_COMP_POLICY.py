import os
import sqlite3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# ────────────────────────────────
# IEEE single-column figure style
# ────────────────────────────────
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

# ────────────────────────────────
# Your AoI helpers (unchanged)
# ────────────────────────────────
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
                                         db_name: str = "slotwise_data.sqlite",
                                         align_first_N: Optional[int] = None) -> Tuple[float, np.ndarray]:
    """
    Return (mean AoI across episodes, full episode series) for a run.
    If align_first_N is given, mean is taken over the first N episodes of the series.
    """
    db_path = os.path.join(run_dir, db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        ser = episode_end_avg_series(conn, num_slots, frames_per_episode, num_episodes)
    finally:
        conn.close()
    if align_first_N is not None:
        N = min(align_first_N, len(ser))
        mean_val = float(np.nanmean(ser[:N]))
    else:
        mean_val = float(np.nanmean(ser))
    return mean_val, ser

# ────────────────────────────────
# Your directory builders (as provided)
# ────────────────────────────────
def build_dir_proposed(u: int, num_slots: int, num_episodes_GTH: int, gamma_th_db: int) -> str:
    base = f"AoI_U{u}_S{num_slots}_GTH"
    sub  = f"AoI_U{u}_S{num_slots}_EP{num_episodes_GTH}_GTH{('minus' + str(abs(gamma_th_db))) if gamma_th_db < 0 else gamma_th_db}"
    return os.path.join(base, sub)

def build_dir_threshold_v2(u: int, num_slots: int, num_episodes: int, gamma_th_db: int) -> str:
    base = f"Threshold User {u}"
    sub  = f"AoI_U{u}_S{num_slots}_EP{num_episodes}TH{('minus' + str(abs(gamma_th_db))) if gamma_th_db < 0 else gamma_th_db}_PolicyV2"
    return os.path.join(base, sub)

def build_dir_greedy(u: int, num_slots: int, num_episodes: int, gamma_th_db: int) -> str:
    base = f"AOI SINR TH GREED U60" #f"AoI_U{u}_S{num_slots}_Greedy"
    sub  = f"AoI_U{u}_S{num_slots}_EP{num_episodes}_Greedy_TH{('minus' + str(abs(gamma_th_db))) if gamma_th_db < 0 else gamma_th_db}"
    return os.path.join(base, sub)

def build_dir_random(u: int, num_slots: int, num_episodes: int, gamma_th_db: int) -> str:
    base = f"AoI_U{u}_S{num_slots}_THRNDM"
    sub  = f"AoI_U{u}_S{num_slots}_EP{num_episodes}_TH{('minus' + str(abs(gamma_th_db))) if gamma_th_db < 0 else gamma_th_db}RNDM"
    return os.path.join(base, sub)

# ────────────────────────────────
# Plot: AoI vs SINR threshold (four policies)
# ────────────────────────────────
def plot_aoi_vs_threshold_four_policies(
    thresholds_db: List[int],
    num_users: int,
    num_slots: int,
    frames_per_episode: int,
    num_episodes: int,        # PPO/Greedy/Random EP horizon
    num_episodes_GTH: int,     # Threshold EP horizon (can differ)
    out_pdf: str = "aoi_vs_threshold_four_policies.pdf",
    align_first_N: Optional[int] = None,  # e.g., set to min(num_episodes, num_episodes_TH)=50 for strict comparability
):
    x_vals = list(thresholds_db)

    def collect_curve(builder, label, ep_count):
        ys = []
        for th in x_vals:
            run_dir = builder(num_users, num_slots, ep_count, int(th))
            try:
                y, _ = mean_of_episode_end_averages_for_run(
                    run_dir, num_slots, frames_per_episode, ep_count,
                    db_name="slotwise_data.sqlite",
                    align_first_N=align_first_N
                )
                print(f"[{label}] th={th:>3} dB → Mean AoI={y:.3f} | dir={run_dir}")
            except Exception as e:
                print(f"[WARN] {label} | th={th}: {e} | dir={run_dir}")
                y = float("nan")
            ys.append(y)
        return np.array(ys, dtype=float)

    # Gather curves
    y_ppo       = collect_curve(build_dir_proposed,     "Proposed PPO", num_episodes_GTH)
    y_random    = collect_curve(build_dir_random,       "Random",       num_episodes_GTH)
    y_greedy    = collect_curve(build_dir_greedy,       "Greedy",       num_episodes)
    y_threshold = collect_curve(build_dir_threshold_v2, "Threshold",    num_episodes)

    # Plot
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.xticks(x_vals, [f"{int(v)}" for v in x_vals])

    # PPO (purple)
    plt.plot(x_vals, y_ppo, color="#6A0DAD", marker="o",
             markerfacecolor="none", markeredgecolor="#6A0DAD",
             markersize=4.8, linewidth=1.4, label="Proposed PPO")

    # Random (gray)
    plt.plot(x_vals, y_random, color="#555555", marker="s",
             markerfacecolor="none", markeredgecolor="#555555",
             markersize=4.8, linewidth=1.3, label="Random")

    # Greedy (green)
    plt.plot(x_vals, y_greedy, color="#2E8B57", marker="^",
             markerfacecolor="none", markeredgecolor="#2E8B57",
             markersize=4.8, linewidth=1.3, label="Greedy")

    # Threshold (orange)
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
    print(f"[PLOT] Saved → {out_pdf}")

    return {
        "thresholds_db": x_vals,
        "Proposed PPO": y_ppo,
        "Random": y_random,
        "Greedy": y_greedy,
        "Threshold": y_threshold,
    }

# ────────────────────────────────
# Example main (your settings)
# ────────────────────────────────
if __name__ == "__main__":
    num_users = 60
    num_slots = 15
    frames_per_episode = 200
    num_episodes     = 100    # PPO/Greedy/Random
    num_episodes_GTH  = 50   # Threshold
    thresholds_db = [-20, -15, -10, -5, 0]

    # Strict comparability (optional): average SAME first N episodes for all
    # align_first_N = min(num_episodes, num_episodes_TH)  # -> 50
    align_first_N = None  # set to 50 if you want strict comparability

    plot_aoi_vs_threshold_four_policies(
        thresholds_db=thresholds_db,
        num_users=num_users,
        num_slots=num_slots,
        frames_per_episode=frames_per_episode,
        num_episodes=num_episodes,
        num_episodes_GTH=num_episodes_GTH,
        out_pdf="aoi_vs_threshold_four_policiesTT.png",
        align_first_N=align_first_N
    )
