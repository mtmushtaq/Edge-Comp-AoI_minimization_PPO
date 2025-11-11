#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final System Average AoI vs. Number of Slots

- Base directory is fixed to the user's OneDrive path (edit BASE_DIR below if needed).
- Expects per-slot runs living under:
      {BASE_DIR}/AoI_U{M_total}_S{num_slots}_SLT
- For each S in slot_list, computes the FINAL system average AoI (the last point of
  the cumulative/running mean on the global slot axis) and plots AAoI vs. S.

It tries to load:
  (1) Per-user AoI telemetry pkl: user_aoi.pkl / aoi_log.pkl / aoi_telemetry.pkl / *aoi*.pkl
      Format: data[uid] = list of dicts {'ep','frame','slot','aoi',...}
  (2) Fallback SAR pkl: sar_logU{M_total}S{num_slots}.pkl
      Assumes state[:,0:2] are (AoI_near, AoI_far).

Outputs:
  - system_aoi_vs_slots.pdf (in BASE_DIR)
  - system_aoi_vs_slots.csv (in BASE_DIR)
"""

import os
import glob
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------- configure your base directory here ----------------------
BASE_DIR = "/Users/muhammadtauseefmushtaq/Library/CloudStorage/OneDrive-PolitecnicodiBari/AOI PPO ICC 2026"

# ---------------------- plotting style (matches your existing plots) ------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
    "mathtext.fontset": "stix", "axes.unicode_minus": False, "pdf.use14corefonts": True,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.8, "lines.linewidth": 1.3, "grid.linewidth": 0.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

# ---------------------- helpers to compute final AAoI ----------------------
def _final_system_aoi_from_user_data(data, num_slots, frames_per_episode=None, num_episodes=None):
    """
    data: dict[uid] -> list of rows, each {'ep','frame','slot','aoi',...}
    Returns final system running mean AoI (float).
    """
    uids = sorted(data.keys())
    if not uids:
        return np.nan

    # If frames_per_episode/num_episodes are unknown, we can still compute a timeline
    # using the max observed composite time index.
    frames_per_episode = frames_per_episode or 1
    num_episodes = num_episodes or 10

    T = num_slots * frames_per_episode
    T_total = T * num_episodes

    sum_per_t, cnt_per_t = {}, {}
    for uid in uids:
        rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        for r in rows:
            t = (r["ep"] - 1) * T + r["frame"] * num_slots + r["slot"]
            aoi = float(r.get("aoi", 0.0))
            sum_per_t[t] = sum_per_t.get(t, 0.0) + aoi
            cnt_per_t[t] = cnt_per_t.get(t, 0) + 1

    if not sum_per_t:
        return np.nan

    t_sorted = np.array(sorted(sum_per_t.keys()), dtype=int)
    max_t = int(min(t_sorted.max(), T_total))

    per_slot_mean = np.full(max_t + 1, np.nan)
    for t in t_sorted:
        if t <= max_t:
            per_slot_mean[t] = sum_per_t[t] / max(cnt_per_t[t], 1)

    # forward fill gaps
    last = np.nan
    for i in range(per_slot_mean.size):
        if np.isfinite(per_slot_mean[i]):
            last = per_slot_mean[i]
        else:
            per_slot_mean[i] = last

    valid = np.isfinite(per_slot_mean)
    ps = np.where(valid, per_slot_mean, 0.0)
    w  = np.where(valid, 1.0, 0.0)
    csum = np.cumsum(ps)
    wsum = np.cumsum(w)
    running_mean = np.divide(csum, np.maximum(wsum, 1e-12))
    return float(running_mean[-1])

def _final_system_aoi_from_sar(states):
    """
    states: array [T, D] with D>=2; columns 0,1 are AoI_near, AoI_far (as in your state).
    Returns final running mean of per-sample mean AoI across near&far.
    """
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] < 2:
        return np.nan
    a_n = states[:, 0].astype(float)
    a_f = states[:, 1].astype(float)
    per_sample_mean = np.nanmean(np.vstack([a_n, a_f]).T, axis=1)
    per_sample_mean = np.nan_to_num(per_sample_mean, nan=np.nanmean(per_sample_mean[np.isfinite(per_sample_mean)]) if np.isfinite(per_sample_mean).any() else 0.0)
    csum = np.cumsum(per_sample_mean)
    idx = np.arange(1, per_sample_mean.size + 1, dtype=float)
    running_mean = csum / idx
    return float(running_mean[-1])

# ---------------------- loader (AoI telemetry first, SAR fallback) -------------
def load_final_aoi_from_run_dir(run_dir, M_total, num_slots,
                                frames_per_episode=None, num_episodes=None,
                                verbose=True):
    """
    Try AoI telemetry pkls; then fall back to SAR states.
    """
    # try AoI telemetry
    candidates = [
        "user_aoi.pkl",
        "aoi_log.pkl",
        "aoi_telemetry.pkl",
    ]
    # also any *aoi*.pkl
    candidates += [os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "*aoi*.pkl"))]

    tried = set()
    for name in candidates:
        path = os.path.join(run_dir, name)
        if path in tried or not os.path.isfile(path):
            continue
        tried.add(path)
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict) and data:
                val = _final_system_aoi_from_user_data(
                    data, num_slots,
                    frames_per_episode=frames_per_episode,
                    num_episodes=num_episodes
                )
                if verbose:
                    print(f"[OK] {os.path.relpath(path, BASE_DIR)} → final AAoI={val:.4f}")
                return val
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to load {path}: {e}")

    # fallback: SAR
    sar_path = os.path.join(run_dir, f"sar_logU{M_total}S{num_slots}.pkl")
    if os.path.isfile(sar_path):
        try:
            with open(sar_path, "rb") as fh:
                sar = pickle.load(fh)
            states = None
            if isinstance(sar, dict):
                for k in sar.keys():
                    if k.lower() in ("s", "state", "states"):
                        states = np.asarray(sar[k])
                        break
            elif isinstance(sar, (list, tuple)) and len(sar) > 0:
                first = sar[0]
                if isinstance(first, (list, tuple)) and len(first) >= 1:
                    states = np.vstack([np.asarray(x[0]) for x in sar])
                elif isinstance(first, dict) and ("s" in first or "state" in first):
                    key = "s" if "s" in first else "state"
                    states = np.vstack([np.asarray(x[key]) for x in sar])

            if states is not None:
                val = _final_system_aoi_from_sar(states)
                if verbose:
                    print(f"[OK] {os.path.relpath(sar_path, BASE_DIR)} (SAR) → final AAoI={val:.4f}")
                return val
            else:
                if verbose:
                    print(f"[WARN] SAR found but no states key in {sar_path}")
        except Exception as e:
            if verbose:
                print(f"[WARN] Failed to load SAR {sar_path}: {e}")
    else:
        if verbose:
            print(f"[MISS] No *aoi*.pkl or SAR in {os.path.relpath(run_dir, BASE_DIR)}")

    return np.nan

# ---------------------- top-level plotting -------------------------------------
def plot_final_aoi_vs_slots(M_total,
                            slot_list=(3, 5, 8, 11, 13),
                            frames_per_episode=None,
                            num_episodes=None,
                            out_pdf="system_aoi_vs_slots.pdf",
                            out_csv="system_aoi_vs_slots.csv"):
    xs, ys = [], []

    for S in slot_list:
        run_dir = os.path.join(BASE_DIR, f"AoI_U{M_total}_S{S}_SLT")
        val = load_final_aoi_from_run_dir(run_dir, M_total, S,
                                          frames_per_episode=frames_per_episode,
                                          num_episodes=num_episodes,
                                          verbose=True)
        xs.append(S)
        ys.append(val)

    # Save CSV to BASE_DIR
    csv_path = os.path.join(BASE_DIR, out_csv)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("slots,final_system_AAoI\n")
        for s, v in zip(xs, ys):
            f.write(f"{s},{('NaN' if not np.isfinite(v) else f'{v:.6f}')}\n")
    print(f"[INFO] CSV saved → {csv_path}")

    # Plot to BASE_DIR
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    mask = np.isfinite(ys)
    if np.any(mask):
        ax.plot(np.array(xs)[mask], np.array(ys)[mask], "-o", ms=4)
    else:
        ax.plot(xs, ys, "-o", ms=4)

    ax.set_xlabel("Number of Slots (N)")
    ax.set_ylabel("Final System Average AoI")
    title = "Final System AAoI vs. Slots"
    if frames_per_episode and num_episodes:
        title += f"  (frames={frames_per_episode}, episodes={num_episodes})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf_path = os.path.join(BASE_DIR, out_pdf)
    fig.savefig(pdf_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] PDF saved → {pdf_path}")

    # Console table
    print("\n=== Final System AAoI by Slots ===")
    for s, v in zip(xs, ys):
        print(f"N={s:<3}  AAoI={('NaN' if not np.isfinite(v) else f'{v:.4f}')}")

# ---------------------- main ---------------------------------------------------
if __name__ == "__main__":
    # Fill these if you want them printed in the title (optional)
    M_TOTAL = 18
    FRAMES_PER_EPISODE = None  # e.g., 1000
    NUM_EPISODES = None        # e.g., 30

    SLOT_LIST = (3, 5, 8, 11, 13)

    plot_final_aoi_vs_slots(M_TOTAL,
                            slot_list=SLOT_LIST,
                            frames_per_episode=FRAMES_PER_EPISODE,
                            num_episodes=NUM_EPISODES)
