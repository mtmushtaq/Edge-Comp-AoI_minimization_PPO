# Utility functions to produce the three plots you described.
# Notes (per your constraints & my plotting rules):
# - Uses matplotlib only (no seaborn).
# - Each chart is its own figure (no subplots).
# - No explicit colors are set EXCEPT the green circles you requested for episode-end markers.
# - Saves figures to PDF in the provided output directory.
#
# Expected input "data" format:
# data: dict[user_id] -> list of records, where record has at least:
#   {"ep": int, "frame": int, "slot": int, "aoi": float}
#
# Other inputs:
#   num_slots: slots per frame (int)
#   frames_per_episode: frames per episode (int)
#   num_episodes: number of episodes (int)
#   out_dir: output folder path (str)

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import os
import re
import csv
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import re
import csv
import math
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F
from pandas.io import json
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

EPS = 1e-8


class Telemetry:
    def __init__(self):
        self.by_uid = defaultdict(list)
        self._tick = 0

    @staticmethod
    @staticmethod
    def _parse_uid(u):
        """
        Accepts a UserState (with .uid like 'U32'), a string 'U32'/'32',
        or an int 32. Returns (uid_num:int, uid_str:str).
        """
        # UserState
        if hasattr(u, "uid"):
            raw = u.uid
        else:
            raw = u

        # string like 'U2' or '2'
        if isinstance(raw, str):
            m = re.search(r"\d+", raw)
            if m:
                n = int(m.group(0))
                return n, f"U{n}"
            else:
                raise ValueError(f"String uid must contain a number: {raw!r}")

        # integer
        if isinstance(raw, (int, np.integer)):
            return int(raw), f"U{int(raw)}"

        raise ValueError(f"Invalid uid: {u!r} (expected UserState/int/'U#')")

    def tick(self):
        self._tick += 1
        return self._tick

    def log_user(self, ep, u, frame, slot, kind, sinr, battery,
                 harvested, required, decoded, aoi, distance,
                 scheduled, pd_role):
        uid_num, uid_str = self._parse_uid(u)
        row = {
            "ep": int(ep),
            "frame": int(frame),
            "slot": int(slot),
            "uid": uid_num,
            "uid_str": uid_str,
            "step": self.tick(),
            "kind": str(kind) if kind else "",
            "pd_role": str(pd_role) if pd_role else "",
            "scheduled": int(bool(scheduled)),
            "decoded": int(bool(decoded)),
            "required": int(bool(required)),
            "aoi": float(aoi) if aoi is not None else 0.0,
            "battery": float(battery) if battery is not None else 0.0,
            "harvested": float(harvested) if harvested is not None else 0.0,
            "sinr": float(sinr) if sinr is not None else 0.0,
            "distance": float(distance) if distance is not None else 0.0,

        }
        self.by_uid[uid_num].append(row)

    def export_frame_csv(self, run_dir, ep, frame, filename="telemetry_rows.csv", warn_missing=True):
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        write_header = not os.path.exists(path)
        missing_ct, written_ct = 0, 0

        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["ep","frame","slot","uid","uid_str","step","kind","pd_role",
                            "scheduled","decoded","required","aoi","battery",
                            "harvested","sinr","distance"])

            for uid_num, rows in list(self.by_uid.items()):
                for r in rows:
                    r_ep = r.get("ep"); r_fr = r.get("frame")
                    if r_ep is None or r_fr is None:
                        missing_ct += 1
                        continue
                    if r_ep == ep and r_fr == frame:
                        w.writerow([
                            r_ep, r_fr, r.get("slot", 0), r.get("uid", uid_num),
                            r.get("uid_str", f"U{uid_num}"), r.get("step", 0),
                            r.get("kind",""), r.get("pd_role",""),
                            int(bool(r.get("scheduled", 0))),
                            int(bool(r.get("decoded", 0))),
                            int(bool(r.get("required", 0))),
                            float(r.get("aoi", 0.0)), float(r.get("battery", 0.0)),
                            float(r.get("harvested", 0.0)), float(r.get("sinr", 0.0)),
                            float(r.get("distance", 0.0))
                        ])
                        written_ct += 1

        if warn_missing and missing_ct:
            print(f"[telemetry] export_frame_csv: skipped {missing_ct} malformed row(s) (ep={ep}, frame={frame}).")
        return written_ct

    def clear_frame(self, ep, frame):
        removed = 0
        for uid_num in list(self.by_uid.keys()):
            rows = self.by_uid[uid_num]
            keep = [r for r in rows if not (r.get("ep") == ep and r.get("frame") == frame)]
            removed += len(rows) - len(keep)
            if keep:
                self.by_uid[uid_num] = keep
            else:
                del self.by_uid[uid_num]
        return removed

    from pathlib import Path
    import numpy as np
    from pathlib import Path

    def save_episode_npy(self, run_dir, filename="episode_data.npy"):
        """
        Save the episode-wise telemetry data (.npy) into the specified run directory.
        """
        import os
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)

        data = dict(self.by_uid)
        np.save(path, data)
        print(f"[SAVE] Saved episode data to {path}")

    def save_slotwise_dat(self, run_dir, filename="slotwise_data.dat"):
        """
        Save complete slotwise raw data (.dat) for later loading.
        """
        import os, pickle
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)  # ✅ build full file path

        with open(path, "wb") as f:
            pickle.dump(self.by_uid, f)

    def save_slotwise_npy(self, run_dir, filename="slotwise_data.npy"):
        """
        Save complete slotwise telemetry data in .npy format.
        """
        import os
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        np.save(path, self.by_uid)
        print(f"[SAVE] Saved slotwise data to {path}")


    def compute_average_aoi(self, total_slots):
        uid_avg = {}

        for uid, rows in self.by_uid.items():
            aoi_sum = sum(r["aoi"] for r in rows)
            uid_avg[uid] = aoi_sum / total_slots
        return uid_avg

    def compute_system_aoi(self, total_slots):
        user_avgs = self.compute_average_aoi(total_slots)

        return sum(user_avgs.values()) / len(user_avgs)

telemetry = Telemetry()
# The remaining logic for state construction, slot assignment, frame rollout,
# .dat and .npy file generation, average AoI tracking, belief model for decoding probability,
# and plotting routines should now be added below step-by-step.

# Confirm once you're ready for the next complete segment: full rollout + .dat/.npy tracking and plots.

class SARLogger:
    def __init__(self):
        self.logs = []

    def log(self, ep, frame, slot, state, action, reward=None):
        """
        Store a single SAR entry for the given slot decision.
        """
        self.logs.append({
            "ep": ep,
            "frame": frame,
            "slot": slot,
            "state": state,       # vector (e.g. numpy array)
            "action": action,     # int slot index
            "reward": reward      # can be filled later
        })

    def set_reward(self, ep, frame, slot, reward):
        """
        After decoding outcome, update the reward for that decision.
        """
        for r in self.logs:
            if r["ep"] == ep and r["frame"] == frame and r["slot"] == slot:
                r["reward"] = reward
                return
        raise ValueError(f"No matching SAR entry for ep={ep}, frame={frame}, slot={slot}")

    def save(self, run_dir, filename="sar_log.pkl"):
        import os, pickle
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(self.logs, f)
        print(f"[SAVE] SAR log saved → {path}")

    def load(self, filepath):
        import pickle
        with open(filepath, "rb") as f:
            self.logs = pickle.load(f)

sar_logger = SARLogger()

def _episode_slot_count(num_slots, frames_per_episode):
    """Total slots per episode (slot index steps per episode)."""
    return num_slots * frames_per_episode

def _episode_end_indices(num_slots, frames_per_episode, num_episodes):
    """Return the global slot indices that correspond to end-of-episode positions (1-based inside an episode → map to running global index)."""
    T = _episode_slot_count(num_slots, frames_per_episode)
    # We'll use 0-based global slot index in plots. End of episode k is at (k*T - 1).
    return [k*T - 1 for k in range(1, num_episodes + 1)]

def _linear_time_index(r, num_slots, frames_per_episode):
    """Map (ep, frame, slot) to a single global time index (0-based)."""
    T = _episode_slot_count(num_slots, frames_per_episode)
    # assuming r["frame"] is 0-based and r["slot"] is 0-based. If yours are 1-based, adjust here.
    return (r["ep"] - 1) * T + r["frame"] * num_slots + r["slot"]

def _sorted_rows_for_user(rows, num_slots, frames_per_episode):
    """Sort rows by (ep, frame, slot) and attach global time index."""
    rows_sorted = sorted(rows, key=lambda rr: (rr["ep"], rr["frame"], rr["slot"]))
    for rr in rows_sorted:
        rr["_t"] = _linear_time_index(rr, num_slots, frames_per_episode)
    return rows_sorted

def _group_indices_by_episode(num_slots, frames_per_episode, num_episodes):
    """Return a list of (start_idx, end_idx_inclusive) index ranges per episode in 0-based global slot coordinates."""
    T = _episode_slot_count(num_slots, frames_per_episode)
    groups = []
    for ep in range(num_episodes):
        start = ep * T
        end   = start + T - 1
        groups.append((start, end))
    return groups

def plot_user_avg_aoi_with_episode_markers_and_variance(
    data, num_slots, frames_per_episode, num_episodes,
    out_dir, out_pdf="Users_AvgAoI_with_EpisodeMarkersAndVar.pdf"
):
    """Plot: per-user running average AoI vs global slot, with GREEN circle markers at the end of each episode.
       At episode ends, draw a vertical error bar using the *variance* of that episode for that user (variance over that user's per-slot AoI values within the episode).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Aesthetics
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    T = _episode_slot_count(num_slots, frames_per_episode)
    episode_ends = _episode_end_indices(num_slots, frames_per_episode, num_episodes)
    ep_ranges = _group_indices_by_episode(num_slots, frames_per_episode, num_episodes)

    uids = sorted(data.keys())

    fig = plt.figure(figsize=(10, 4.0))
    ax = fig.add_subplot(111)
    ax.set_title("Per-User Running Average AoI (Green circles: episode ends)")
    ax.set_xlabel("Global Slot Index")
    ax.set_ylabel("Average AoI")
    ax.grid(True, alpha=0.3)

    for uid in uids:
        rows = _sorted_rows_for_user(data[uid], num_slots, frames_per_episode)
        if not rows:
            continue

        # Build time series
        ts = [r["_t"] for r in rows]
        aois = np.array([float(r.get("aoi", 0.0)) for r in rows], dtype=float)

        # Running average over global time
        cum = np.cumsum(aois)
        avg = cum / (np.arange(len(aois)) + 1)

        # Plot the line for this user (default line color)
        ax.plot(ts, avg, label=f"U{uid}", linewidth=1.2)

        # Compute per-episode variance for this user (over AOI values within that episode)
        ep_vars = []
        ep_end_y = []
        ep_end_x = []
        for (start, end) in ep_ranges:
            if end >= len(avg):
                # If data shorter than full episodes (defensive), break
                break
            # Episode's raw AoIs for the user:
            seg_vals = aois[start:end+1]
            if seg_vals.size == 0:
                continue
            var_val = float(np.var(seg_vals, ddof=0))  # population variance over that episode
            ep_vars.append(var_val)

            # Episode end marker values (running avg at the end index)
            ep_end_x.append(end)
            ep_end_y.append(float(avg[end]))

        # GREEN hollow circles at episode ends
        if ep_end_x:
            ax.plot(
                ep_end_x, ep_end_y,
                "o", markerfacecolor="none", markeredgecolor="green", markersize=5,
                linestyle="None"
            )

        # Plot variance at episode ends as vertical error bars (using variance directly as "error" height).
        # Since errorbar uses +/- yerr, we set yerr as the variance (not std), to match your request literally.
        if ep_end_x and ep_vars:
            yerr = np.array(ep_vars, dtype=float)
            # Ensure positive:
            yerr[yerr < 0] = 0.0
            ax.errorbar(ep_end_x, ep_end_y, yerr=yerr, fmt="none", capsize=2, elinewidth=0.8)

    ax.set_xlim(0, T * num_episodes)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved per-user Avg AoI with episode markers & variance → {out_path}")

# Replace/define `plot_system_avg_aoi_with_episode_markers_and_variance` so it
# computes the **system average AoI** as the mean of users' **moving averages**
# at each time t, then marks episode ends with green circles and draws per-episode
# variance bars of the system curve.
#
# This matches your intended call:
# plot_system_avg_aoi_with_episode_markers_and_variance(
#     data, num_slots, frames_per_episode, num_episodes, out_dir=plot_dir
# )
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def _episode_slot_count(num_slots, frames_per_episode):
    return num_slots * frames_per_episode

def _sorted_rows_for_user(rows, num_slots, frames_per_episode):
    rows_sorted = sorted(rows, key=lambda rr: (rr["ep"], rr["frame"], rr["slot"]))
    T = _episode_slot_count(num_slots, frames_per_episode)
    for rr in rows_sorted:
        # ep is 1-based; frame/slot are 0-based per your loops
        rr["_t"] = (rr["ep"] - 1) * T + rr["frame"] * num_slots + rr["slot"]
    return rows_sorted

def compute_user_moving_avgs(data, num_slots, frames_per_episode, num_episodes):
    out = {}
    T = _episode_slot_count(num_slots, frames_per_episode)
    total_len = T * num_episodes
    for uid, rows in data.items():
        rows_sorted = _sorted_rows_for_user(rows, num_slots, frames_per_episode)
        if not rows_sorted:
            out[uid] = {"t": np.array([], dtype=int), "mavg": np.array([], dtype=float)}
            continue
        t = np.array([rr["_t"] for rr in rows_sorted], dtype=int)
        a = np.array([float(rr.get("aoi", 0.0)) for rr in rows_sorted], dtype=float)
        # keep only 0..total_len-1 indices
        mask = (t >= 0) & (t < total_len)
        t = t[mask]; a = a[mask]
        if t.size == 0:
            out[uid] = {"t": np.array([], dtype=int), "mavg": np.array([], dtype=float)}
            continue
        c = np.cumsum(a)
        denom = np.arange(1, a.size + 1, dtype=float)
        mavg = c / denom
        out[uid] = {"t": t, "mavg": mavg}
    return out

def plot_system_avg_aoi_with_episode_markers_and_variance(
    data,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir,
    out_pdf="System_AvgAoI_with_EpisodeMarkersAndVar.pdf",
    rolling_window=None,            # e.g., 1000 for smoothing
    use_stddev_instead_of_variance=False,  # True => plot std-dev bars (same units as AoI)
):
    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    T = _episode_slot_count(num_slots, frames_per_episode)
    total_len = T * num_episodes
    uids = sorted(data.keys())
    U = len(uids)

    # --- Compute per-user moving averages ---
    user_series = compute_user_moving_avgs(data, num_slots, frames_per_episode, num_episodes)

    # Align to common timeline [0..total_len-1] and forward-fill each user
    M = np.full((U, total_len), np.nan, dtype=float)
    for ui, uid in enumerate(uids):
        ts = user_series[uid]["t"]
        mv = user_series[uid]["mavg"]
        if ts.size == 0:
            continue
        row = np.full(total_len, np.nan, dtype=float)
        row[ts] = mv
        last = np.nan
        for i in range(total_len):
            if np.isfinite(row[i]):
                last = row[i]
            else:
                row[i] = last
        M[ui] = row

    # --- System average AoI: mean across users of their moving averages at each t ---
    system_mean = np.nanmean(M, axis=0)

    # Optional rolling smoother
    roll_curve = None
    if rolling_window and rolling_window > 1:
        k = int(rolling_window)
        kernel = np.ones(k) / k
        roll_curve = np.convolve(system_mean, kernel, mode="same")

    # --- Plot ---
    fig = plt.figure(figsize=(10, 3.6))
    ax = fig.add_subplot(111)
    ax.set_title("System Average AoI (Mean of Users' Moving Averages)")
    ax.set_xlabel("Global Slot Index")
    ax.set_ylabel("AoI")
    ax.grid(True, alpha=0.3)
    # Avoid '1e6' scientific notation/offsets
    ax.ticklabel_format(style='plain', axis='x', useOffset=False)
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)

    x = np.arange(total_len, dtype=int)
    ax.plot(x, system_mean, linewidth=1.5, label="System avg AoI (mean of user MAvgs)")
    if roll_curve is not None:
        ax.plot(x, roll_curve, "--", linewidth=1.2, label=f"Rolling mean (w={rolling_window})")

    # Episode-end green circles + per-episode variance (or stddev) bars of the system curve
    ep_end_x, ep_end_y, ep_err = [], [], []
    for ep in range(num_episodes):
        start = ep * T
        end = min(start + T - 1, total_len - 1)
        if end < start:
            continue
        seg = system_mean[start:end + 1]
        if seg.size == 0:
            continue
        ep_end_x.append(end)
        ep_end_y.append(system_mean[end])
        if use_stddev_instead_of_variance:
            ep_err.append(float(np.std(seg, ddof=0)))
        else:
            ep_err.append(float(np.var(seg, ddof=0)))

    if ep_end_x:
        ax.plot(ep_end_x, ep_end_y,
                "o", markerfacecolor="none", markeredgecolor="green", markersize=5,
                linestyle="None")
        if ep_err:
            yerr = np.array(ep_err, dtype=float)
            yerr[yerr < 0] = 0.0
            ax.errorbar(ep_end_x, ep_end_y, yerr=yerr, fmt="none",
                        capsize=2, elinewidth=0.8)  # default line color; green marker remains

    ax.set_xlim(0, total_len - 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] System Avg AoI with episode markers & variance saved → {out_path}")


def plot_aoi_cdf(
    data, num_slots, frames_per_episode, num_episodes,
    out_dir, out_pdf="AOI_Empirical_CDF.pdf",
    show_ccdf=False,            # if True, also draw 1 - F(x) on same axes (dashed)
    show_percentiles=(50, 90, 95, 99),  # None to disable; marks verticals at these %s
):
    """
    Empirical CDF of AoI across ALL users and ALL slots of ALL episodes.
    - Right-continuous step plot (ECDF).
    - Ignores NaN/Inf.
    - Optionally overlays CCDF = 1 - F(x) and percentile markers.
    """
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    # Collect all AoI samples
    vals = []
    T = num_slots * frames_per_episode
    for uid, rows in data.items():
        # uses ep 1-based; frame/slot 0-based, but we only need AoI values here
        for r in sorted(rows, key=lambda rr: (rr["ep"], rr["frame"], rr["slot"])):
            vals.append(r.get("aoi", 0.0))

    if not vals:
        print("[PLOT] No AoI data to plot CDF.")
        return

    x = np.asarray(vals, dtype=float)
    x = x[np.isfinite(x)]               # drop NaN/inf
    if x.size == 0:
        print("[PLOT] No finite AoI values to plot CDF.")
        return

    # Sort and build ECDF (right-continuous step function)
    x.sort()
    n = x.size
    y = np.arange(1, n + 1, dtype=float) / n

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    ax.set_title("Empirical CDF of AoI (All Users & Slots)")
    ax.set_xlabel("AoI")
    ax.set_ylabel("F(x) = P{AoI ≤ x}")
    ax.grid(True, alpha=0.3)

    # Proper ECDF step plot (right-continuous)
    ax.step(x, y, where="post", linewidth=1.3)

    # Optional CCDF overlay (tail)
    if show_ccdf:
        ax.step(x, 1.0 - y, where="post", linestyle="--", linewidth=1.1, label="1 - F(x)")

    # Optional percentile markers
    if show_percentiles:
        for p in show_percentiles:
            if 0 < p < 100:
                xp = np.percentile(x, p)
                ax.axvline(xp, linewidth=0.8, alpha=0.4)
                ax.text(xp, 0.02, f"P{p}≈{xp:.2f}", rotation=90, va="bottom", ha="right", fontsize=7)

    if show_ccdf:
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved AOI empirical CDF → {out_path}")


def plot_system_mean_of_user_mavgs_timewise(
    data,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir,
    out_pdf="System_MeanOfUserMAvgs_Timewise.pdf",
    rolling_window=None,
    mark_episode_ends=True,
    show_variance_bars=True,
):
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    T = num_slots * frames_per_episode
    total_len = T * num_episodes
    uids = sorted(data.keys())
    U = len(uids)

    # --- Compute per-user moving averages ---
    user_series = compute_user_moving_avgs(data, num_slots, frames_per_episode, num_episodes)

    # Align & forward-fill
    M = np.full((U, total_len), np.nan, dtype=float)
    for ui, uid in enumerate(uids):
        ts = user_series[uid]["t"]
        mv = user_series[uid]["mavg"]
        if ts.size == 0:
            continue
        row = np.full(total_len, np.nan, dtype=float)
        row[ts] = mv
        last = np.nan
        for i in range(total_len):
            if np.isfinite(row[i]):
                last = row[i]
            else:
                row[i] = last
        M[ui] = row

    # --- System curve ---
    system_mean = np.nanmean(M, axis=0)

    # Optional rolling mean
    roll_curve = None
    if rolling_window and rolling_window > 1:
        k = int(rolling_window)
        kernel = np.ones(k) / k
        roll_curve = np.convolve(system_mean, kernel, mode="same")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_title("System AoI: Mean of Users' Moving Averages Over Time")
    ax.set_xlabel("Global Slot Index")
    ax.set_ylabel("AoI")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='plain', axis='x', useOffset=False)
    ax.ticklabel_format(style='plain', axis='y', useOffset=False)

    x = np.arange(total_len)
    ax.plot(x, system_mean, linewidth=1.5, label="Mean of users' moving avgs")
    if roll_curve is not None:
        ax.plot(x, roll_curve, "--", linewidth=1.2, label=f"Rolling mean (w={rolling_window})")

    # --- Episode ends, variance bars ---
    if mark_episode_ends:
        ep_end_x, ep_end_y, ep_vars = [], [], []
        for ep in range(num_episodes):
            start = ep * T
            end = min(start + T - 1, total_len - 1)
            seg = system_mean[start:end + 1]
            if seg.size == 0:
                continue
            ep_end_x.append(end)
            ep_end_y.append(system_mean[end])
            ep_vars.append(np.var(seg, ddof=0))  # use np.std(seg) for std-deviation instead

        # Green markers
        ax.plot(ep_end_x, ep_end_y, "o", markerfacecolor="none",
                markeredgecolor="green", markersize=5, linestyle="None")

        # Variance as vertical error bars
        if show_variance_bars:
            yerr = np.array(ep_vars, dtype=float)
            yerr[yerr < 0] = 0
            ax.errorbar(ep_end_x, ep_end_y, yerr=yerr, fmt="none",
                        capsize=2, elinewidth=0.8, ecolor="green")

    ax.set_xlim(0, total_len - 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] System mean-of-user-mavgs with variance markers saved → {out_path}")


def make_run_dir(M_total, num_slots, num_episodes):
    #stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"AoI_U{M_total}_S{num_slots}_LR3e32"
    out = os.path.join(name)
    os.makedirs(out, exist_ok=True)
    return out

def load_episode_telemetry(run_dir, filename="episode_data.npy"):
    """
    Load telemetry data from a .npy file stored in a run directory.
    """
    import os
    path = os.path.join(run_dir, filename)
    print(f"[LOAD] Loading episode data from {path}")
    return np.load(path, allow_pickle=True).item()


import os
from typing import Dict, Sequence, Tuple, Optional, Iterable, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- Matplotlib defaults (lightweight, PDF-friendly) ----------
mpl.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "pdf.use14corefonts": True,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 120,
})


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x.astype(float, copy=True)
    # Use cumulative sum for O(n) rolling mean without big temp arrays
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = (c[w:] - c[:-w]) / float(w)
    # pad to original length (left-pad with first value to keep alignment intuitive)
    pad = np.full(w - 1, out[0] if out.size > 0 else np.nan, dtype=float)
    return np.concatenate([pad, out])


def plot_empirical_cdf_streaming(
    samples: Union[np.ndarray, Iterable[float]],
    out_path: str,
    nbins: int = 2048,
    xlim: Optional[Tuple[float, float]] = None,
    title: str = "Empirical CDF",
) -> None:
    """
    Memory-safe empirical CDF via histogram accumulation.
    - 'samples' can be a big 1D numpy array or any iterable of numbers.
    - Avoids sorting all points in memory (which can trigger OOM for tens of millions of values).
    """
    # First pass: get min/max and count to define bins
    min_x, max_x, n = np.inf, -np.inf, 0
    if isinstance(samples, np.ndarray):
        if samples.size == 0:
            raise ValueError("No samples for CDF.")
        min_x = float(np.nanmin(samples))
        max_x = float(np.nanmax(samples))
        n = int(np.isfinite(samples).sum())
        iterable = [samples]  # single chunk
    else:
        # streaming
        buf = []
        for v in samples:
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(vf):
                continue
            buf.append(vf)
            if vf < min_x: min_x = vf
            if vf > max_x: max_x = vf
            n += 1
        iterable = [np.array(buf, dtype=float)]
        if n == 0:
            raise ValueError("No numeric/finite samples for CDF.")

    # Edge case: all equal
    if max_x == min_x:
        edges = np.linspace(min_x - 0.5, max_x + 0.5, nbins + 1)
    else:
        edges = np.linspace(min_x, max_x, nbins + 1)

    # Second pass: histogram accumulation
    counts = np.zeros(nbins, dtype=np.int64)
    for chunk in iterable:
        hist, _ = np.histogram(chunk, bins=edges)
        counts += hist

    cdf = np.cumsum(counts).astype(float)
    cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
    xs = 0.5 * (edges[:-1] + edges[1:])

    _ensure_dir(out_path)
    plt.figure()
    plt.plot(xs, cdf, linewidth=1.0)
    plt.xlabel("AoI")
    plt.ylabel("F(AoI \u2264 x)")
    plt.title(title)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_user_mavg_with_episode_variance(
    user_id: str,
    aoi_series: np.ndarray,
    episode_len: int,
    num_episodes: int,
    out_path: str,
    mavg_window: int = 100,
    skip_first_episode_var: bool = True,
    y_min: float = 300.0,
    y_max: Optional[float] = None,
    annotate_text: bool = False,
) -> None:
    """
    Plot a single user's AoI moving average over global time, and mark the per-episode variance
    at the last slot of each episode.
    - aoi_series: 1D array of length == episode_len * num_episodes, slot-wise over global time.
    - episode variance: var of raw AoI within that episode (not the moving average), shown as a marker.
    """
    assert aoi_series.ndim == 1, "aoi_series must be 1D"
    assert aoi_series.size == episode_len * num_episodes, "len(aoi_series) must be episode_len * num_episodes"

    mavg = moving_average(aoi_series, mavg_window)

    # Compute per-episode variance of raw AoI
    variances = []
    episode_end_indices = []
    for ep in range(num_episodes):
        start = ep * episode_len
        end = start + episode_len
        var = float(np.var(aoi_series[start:end], ddof=0))
        variances.append(var)
        episode_end_indices.append(end - 1)

    _ensure_dir(out_path)
    plt.figure()
    plt.plot(mavg, linewidth=0.8, label=f"User {user_id} moving avg (w={mavg_window})")
    # episode boundary vertical lines
    for ep in range(1, num_episodes):
        x = ep * episode_len - 0.5
        plt.axvline(x, linewidth=0.5, linestyle="--", alpha=0.5)

    # Plot variance markers (skip first ep if requested)
    start_ep = 1 if skip_first_episode_var and num_episodes > 0 else 0
    xs, ys = [], []
    for ep in range(start_ep, num_episodes):
        idx = episode_end_indices[ep]
        xs.append(idx)
        ys.append(mavg[idx])
    plt.scatter(xs, ys, s=10, marker="o", label="Episode end")
    if annotate_text:
        for ep in range(start_ep, num_episodes):
            idx = episode_end_indices[ep]
            plt.annotate(f"var={variances[ep]:.1f}", (idx, mavg[idx]), xytext=(3, 3),
                         textcoords="offset points", fontsize=6)

    plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)
    plt.xlabel("Global Slot Index")
    plt.ylabel("AoI")
    plt.title(f"User {user_id}: Moving Average AoI with Episode Variance Markers")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_system_average_with_episode_variance(
    users_aoi: Dict[str, np.ndarray],
    episode_len: int,
    num_episodes: int,
    out_path: str,
    mavg_window: int = 100,
    skip_first_episode_var: bool = True,
    y_min: float = 300.0,
    y_max: Optional[float] = None,
) -> None:
    """
    System average AoI over time = mean of users' moving averages at each slot.
    Episode variance marker = variance of the system average within that episode.
    """
    # stack users into (n_users, T)
    series = []
    for k, v in users_aoi.items():
        assert v.ndim == 1 and v.size == episode_len * num_episodes, f"{k} series must be 1D of expected length"
        series.append(v.astype(float))
    A = np.vstack(series)  # (U, T)

    # moving avg per user then average across users
    mavg_users = np.vstack([moving_average(A[i], mavg_window) for i in range(A.shape[0])])
    sys_avg = np.mean(mavg_users, axis=0)

    # per-episode variance of system average (raw, not moving-avg) OR of sys_avg?
    # Following your "end-of-episode marker" idea: compute variance of raw system average within episode.
    raw_sys_avg = np.mean(A, axis=0)
    variances = []
    episode_end_indices = []
    for ep in range(num_episodes):
        start = ep * episode_len
        end = start + episode_len
        variances.append(float(np.var(raw_sys_avg[start:end], ddof=0)))
        episode_end_indices.append(end - 1)

    _ensure_dir(out_path)
    plt.figure()
    plt.plot(sys_avg, linewidth=0.9, label=f"System avg AoI (mean of user MAvgs, w={mavg_window})")

    # episode separators
    for ep in range(1, num_episodes):
        x = ep * episode_len - 0.5
        plt.axvline(x, linewidth=0.5, linestyle="--", alpha=0.5)

    # variance markers (skip first ep if requested)
    start_ep = 1 if skip_first_episode_var and num_episodes > 0 else 0
    xs, ys = [], []
    for ep in range(start_ep, num_episodes):
        idx = episode_end_indices[ep]
        xs.append(idx)
        ys.append(sys_avg[idx])
    plt.scatter(xs, ys, s=12, marker="o", label="Episode end")

    plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)
    plt.xlabel("Global Slot Index")
    plt.ylabel("AoI")
    plt.title("System Average AoI (Mean of Users' Moving Averages) with Episode Variance Markers")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ------------------------------
# Example usage (uncomment to test quickly with synthetic data)
# ------------------------------
# if __name__ == "__main__":
#     U = 5
#     frames_per_episode = 100
#     slots_per_frame = 5
#     episode_len = frames_per_episode * slots_per_frame
#     num_episodes = 20
#     T = episode_len * num_episodes
#     rng = np.random.default_rng(42)
#     users = {}
#     for u in range(U):
#         # create AoI-ish random walk bounded away from 0
#         x = np.cumsum(rng.integers(-2, 6, size=T))
#         x = np.clip(x + 350 + 10*u, 200, None)
#         users[f"U{u+1}"] = x.astype(float)
#
#     # Per-user plots
#     for uid, series in users.items():
#         plot_user_mavg_with_episode_variance(
#             uid, series, episode_len, num_episodes,
#             out_path=f"./telemetry_plots/{uid}_mavg_var.pdf",
#             mavg_window=100, skip_first_episode_var=True, y_min=300
#         )
#
#     # System plot
#     plot_system_average_with_episode_variance(
#         users, episode_len, num_episodes,
#         out_path="./telemetry_plots/system_mavg_var.pdf",
#         mavg_window=100, skip_first_episode_var=True, y_min=300
#     )
#
#     # CDF
#     all_samples = np.concatenate(list(users.values()))
#     plot_empirical_cdf_streaming(all_samples, out_path="./telemetry_plots/aoi_empirical_cdf.pdf", nbins=4096)


def plot_all_users_mavg_with_episode_variance(
    users_aoi: Dict[str, np.ndarray],
    episode_len: int,
    num_episodes: int,
    out_path: str,
    mavg_window: int = 100,
    skip_first_episode_var: bool = True,
    y_min: float = 300.0,
    y_max: Optional[float] = None,
    user_ids: Optional[Sequence[str]] = None,
    downsample: Optional[int] = None,
    annotate_text: bool = False,
) -> None:
    """
    Plot moving-average AoI for multiple users in ONE plot, with each user's episode-end variance markers.
    This function iterates users one-by-one to keep memory usage low (no big stacks).

    Args:
        users_aoi: dict user_id -> 1D AoI np.ndarray of length == episode_len * num_episodes
        episode_len: slots per episode (frames_per_episode * slots_per_frame)
        num_episodes: number of episodes
        out_path: path to save the figure
        mavg_window: moving average window size
        skip_first_episode_var: if True, skip variance marker for episode 1
        y_min, y_max: y-limits (set y_min=300 by default per request)
        user_ids: optional subset list of user IDs to include; if None, include all keys
        downsample: if set to integer k>=2, plot every k-th point to reduce density
        annotate_text: if True, annotate variance values next to end markers (can clutter with many users)
    """
    if user_ids is None:
        user_ids = list(users_aoi.keys())

    # Pre-create figure
    _ensure_dir(out_path)
    plt.figure()

    # Draw episode separators once
    for ep in range(1, num_episodes):
        x = ep * episode_len - 0.5
        plt.axvline(x, linewidth=0.4, linestyle="--", alpha=0.4)

    start_ep = 1 if skip_first_episode_var and num_episodes > 0 else 0

    # Iterate users sequentially
    for uid in user_ids:
        series = users_aoi[uid]
        assert series.ndim == 1 and series.size == episode_len * num_episodes, f"{uid} series length mismatch"

        # Compute moving average (O(T))
        mavg = moving_average(series.astype(float), mavg_window)

        # Optional downsample for plotting density
        if downsample is not None and downsample >= 2:
            mavg_plot = mavg[::downsample]
            xline = np.arange(mavg.size)[::downsample]
        else:
            mavg_plot = mavg
            xline = np.arange(mavg.size)

        plt.plot(xline, mavg_plot, linewidth=0.8, label=f"{uid} (w={mavg_window})")

        # Episode-end indices and per-episode variance (raw)
        xs, ys = [], []
        for ep in range(start_ep, num_episodes):
            end_idx = (ep + 1) * episode_len - 1
            xs.append(end_idx)
            ys.append(mavg[end_idx])

            if annotate_text:
                # variance of *raw* AoI in this episode
                st = ep * episode_len
                en = st + episode_len
                var = float(np.var(series[st:en], ddof=0))
                plt.annotate(f"{uid}: var={var:.1f}", (end_idx, mavg[end_idx]),
                             xytext=(3, 3), textcoords="offset points", fontsize=5)

        plt.scatter(xs, ys, s=8, marker="o")  # markers per user

    plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)
    plt.xlabel("Global Slot Index")
    plt.ylabel("AoI")
    plt.title("All Users: Moving Average AoI with Episode-End Variance Markers")
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_all_users_mavg_with_episode_variance(
    users_aoi: Dict[str, np.ndarray],
    episode_len: int,
    num_episodes: int,
    out_path: str,
    user_ids: Optional[Sequence[str]] = None,
    mavg_window: int = 100,
    skip_first_episode_var: bool = True,
    y_min: float = 300.0,
    y_max: Optional[float] = None,
    decimate: int = 1,
    draw_episode_separators: bool = True,
    show_legend: bool = True,
    title: Optional[str] = None,
) -> None:
    """
    Plot all users in a SINGLE figure, one line per user (moving average),
    with per-episode variance markers at the END of each episode for each user.
    Implemented in a memory-conscious way: it processes each user sequentially
    without stacking big arrays.
    - users_aoi: dict {user_id -> 1D AoI series of length episode_len * num_episodes}
    - decimate: plot every k-th point to reduce rendering load (default 1 = no decimation)
    """
    assert isinstance(users_aoi, dict) and len(users_aoi) > 0, "users_aoi must be a non-empty dict"
    if user_ids is None:
        user_ids = list(users_aoi.keys())

    # Prepare figure
    _ensure_dir(out_path)
    plt.figure()

    # Draw episode separators once (lightweight)
    if draw_episode_separators:
        for ep in range(1, num_episodes):
            x = ep * episode_len - 0.5
            plt.axvline(x, linewidth=0.5, linestyle="--", alpha=0.4)

    # Process each user sequentially
    for uid in user_ids:
        series = users_aoi[uid]
        assert series.ndim == 1 and series.size == episode_len * num_episodes, (
            f"{uid} series must be 1D and length == episode_len * num_episodes"
        )

        # Moving average (computed and immediately plotted to avoid storing many arrays)
        mavg = moving_average(series, mavg_window)

        # Optional decimation for rendering performance
        if decimate > 1:
            # Build x for decimated display
            xs = np.arange(mavg.size)[::decimate]
            plt.plot(xs, mavg[::decimate], linewidth=0.8, label=f"{uid}")
        else:
            plt.plot(mavg, linewidth=0.8, label=f"{uid}")

        # Episode-end variance markers for this user (small arrays only)
        start_ep = 1 if skip_first_episode_var and num_episodes > 0 else 0
        end_indices = [ep * episode_len + (episode_len - 1) for ep in range(num_episodes)]
        if start_ep < num_episodes:
            sel_idx = end_indices[start_ep:]
            sel_y = [mavg[i] for i in sel_idx]
            # Compute variances (of raw AoI within each episode)
            # (We don't necessarily annotate values, just show marker at episode boundary)
            # If later needed, we can annotate with the variance string.
            plt.scatter(sel_idx, sel_y, s=9, marker="o")

        # Free mavg reference ASAP (helps the GC)
        del mavg

    plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)
    plt.xlabel("Global Slot Index")
    plt.ylabel("AoI")
    if title is None:
        title = "All Users: Moving Average AoI with Episode-End Variance Markers"
    plt.title(title)
    if show_legend:
        plt.legend(loc="best", ncol=2, fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ========= Slotwise telemetry loaders & helpers =========

import json
import os

try:
    import torch  # optional; guarded below
except Exception:
    torch = None


def load_episode_telemetry(run_dir: str, filename: str = "episode_data.npy"):
    """Load telemetry data (dict-like) from .npy file stored in run_dir."""
    path = os.path.join(run_dir, filename)
    print(f"[LOAD] Loading episode data from {path}")
    return np.load(path, allow_pickle=True).item()


def _episode_slot_count(num_slots: int, frames_per_episode: int) -> int:
    return int(num_slots) * int(frames_per_episode)


def _sorted_rows_for_user(rows, num_slots: int, frames_per_episode: int):
    """Rows are dicts expected to contain at least keys '_t' and 'aoi'."""
    # Sort by global slot index '_t'
    try:
        return sorted(rows, key=lambda r: int(r.get("_t", 0)))
    except Exception:
        # best-effort fallback
        return rows


def compute_user_moving_avgs(data, num_slots: int, frames_per_episode: int, num_episodes: int):
    """Compute cumulative moving average for each user over its observed slots.
    Returns dict: {uid: {'t': int array, 'mavg': float array}}"""
    out = {}
    T = _episode_slot_count(num_slots, frames_per_episode)
    total_len = T * num_episodes
    for uid, rows in data.items():
        rows_sorted = _sorted_rows_for_user(rows, num_slots, frames_per_episode)
        if not rows_sorted:
            out[uid] = {"t": np.array([], dtype=int), "mavg": np.array([], dtype=float)}
            continue
        t = np.array([int(rr.get("_t", -1)) for rr in rows_sorted], dtype=int)
        a = np.array([float(rr.get("aoi", np.nan)) for rr in rows_sorted], dtype=float)
        mask = (t >= 0) & (t < total_len) & np.isfinite(a)
        t = t[mask]; a = a[mask]
        if t.size == 0:
            out[uid] = {"t": np.array([], dtype=int), "mavg": np.array([], dtype=float)}
            continue
        c = np.cumsum(a)
        denom = np.arange(1, a.size + 1, dtype=float)
        mavg = c / denom
        out[uid] = {"t": t, "mavg": mavg}
    return out


# ========= NaN-aware moving average for sparse slotwise records =========

def moving_average_nan(x: np.ndarray, w: int) -> np.ndarray:
    """NaN-aware rolling mean with fixed window w over a dense vector.
    Uses convolution over a finite mask to ignore NaNs."""
    x = x.astype(float, copy=False)
    n = x.size
    if w <= 1 or n == 0:
        return x.copy()
    mask = np.isfinite(x).astype(float)
    x0 = np.where(np.isfinite(x), x, 0.0)
    # Simple convolution with ones
    kern = np.ones(w, dtype=float)
    num = np.convolve(x0, kern, mode="full")[:n]
    den = np.convolve(mask, kern, mode="full")[:n]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    # For indices with no finite values in window, keep NaN
    out[den == 0] = np.nan
    # Left-edge behavior: we keep 'same-length' output; if you prefer centered, adjust as needed.
    return out


# ========= Plotting from slotwise dict data (memory-safe, per-user sequential) =========

def plot_all_users_from_slotwise_records(
    data: dict,
    episode_len: int,
    num_episodes: int,
    out_path: str,
    user_ids: Optional[Sequence[str]] = None,
    mavg_window: int = 100,
    skip_first_episode_var: bool = True,
    y_min: float = 300.0,
    y_max: Optional[float] = None,
    decimate: int = 1,
    draw_episode_separators: bool = True,
    show_legend: bool = True,
    title: Optional[str] = None,
    annotate_variance: bool = False,
) -> None:
    """
    Plot all users' moving averages in ONE figure from slotwise telemetry dict:
    data: { uid: [ {"_t": <global_slot>, "aoi": <value>, ...}, ... ] }
    Memory-friendly: processes one user at a time without stacking.
    """
    assert isinstance(data, dict) and len(data) > 0, "data must be a non-empty dict"
    if user_ids is None:
        user_ids = list(data.keys())
    T = episode_len * num_episodes

    _ensure_dir(out_path)
    plt.figure()

    if draw_episode_separators:
        for ep in range(1, num_episodes):
            x = ep * episode_len - 0.5
            plt.axvline(x, linewidth=0.5, linestyle="--", alpha=0.4)

    end_indices = [ep * episode_len + (episode_len - 1) for ep in range(num_episodes)]
    start_ep = 1 if skip_first_episode_var and num_episodes > 0 else 0

    for uid in user_ids:
        rows = _sorted_rows_for_user(data[uid], episode_len, episode_len)  # params unused inside
        # Dense vector for this user (NaNs where missing)
        u = np.full(T, np.nan, dtype=float)
        if rows:
            t_idx = np.array([int(r.get("_t", -1)) for r in rows], dtype=int)
            aoi  = np.array([float(r.get("aoi", np.nan)) for r in rows], dtype=float)
            m = (t_idx >= 0) & (t_idx < T) & np.isfinite(aoi)
            u[t_idx[m]] = aoi[m]

        # Moving average ignoring NaNs
        mavg = moving_average_nan(u, mavg_window)

        # Plot line (optionally decimate for rendering)
        if decimate > 1:
            xs = np.arange(T)[::decimate]
            plt.plot(xs, mavg[::decimate], linewidth=0.8, label=f"{uid}")
        else:
            plt.plot(mavg, linewidth=0.8, label=f"{uid}")

        # Episode-end variance markers for THIS user using raw AoI per-episode
        if start_ep < num_episodes:
            sel_idx = end_indices[start_ep:]
            # y from the mavg curve
            sel_y = [mavg[i] for i in sel_idx]
            plt.scatter(sel_idx, sel_y, s=9, marker="o")
            if annotate_variance:
                for ep in range(start_ep, num_episodes):
                    s = ep * episode_len
                    e = s + episode_len
                    var = float(np.nanvar(u[s:e]))
                    plt.annotate(f"v={var:.0f}", (e-1, mavg[e-1]), xytext=(3, 3),
                                 textcoords="offset points", fontsize=6)

        # Free per-user buffers
        del u, mavg

    plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)
    plt.xlabel("Global Slot Index")
    plt.ylabel("AoI")
    if title is None:
        title = "All Users: MAvg AoI (slotwise records) with Episode-End Variance Markers"
    plt.title(title)
    if show_legend:
        plt.legend(loc="best", ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_system_average_from_slotwise_records(
    data: dict,
    episode_len: int,
    num_episodes: int,
    out_path: str,
    mavg_window: int = 100,
    skip_first_episode_var: bool = True,
    y_min: float = 300.0,
    y_max: Optional[float] = None,
    decimate: int = 1,
    title: Optional[str] = None,
) -> None:
    """
    Compute and plot the system average over time from slotwise dict records, in a memory-aware way:
    - System curve = mean across users of EACH USER'S moving average (window mavg_window).
    - Episode variance markers computed on the raw system average within each episode.
    Implementation: iterate users, update running sums to avoid stacking (U,T).
    """
    T = episode_len * num_episodes
    # Accumulate sum and count for moving-average curves
    sum_mavg = np.zeros(T, dtype=float)
    cnt_mavg = np.zeros(T, dtype=float)
    # Also accumulate raw AoI to compute raw system average later
    sum_raw  = np.zeros(T, dtype=float)
    cnt_raw  = np.zeros(T, dtype=float)

    user_ids = list(data.keys())

    for uid in user_ids:
        rows = _sorted_rows_for_user(data[uid], episode_len, episode_len)
        u = np.full(T, np.nan, dtype=float)
        if rows:
            t_idx = np.array([int(r.get("_t", -1)) for r in rows], dtype=int)
            aoi  = np.array([float(r.get("aoi", np.nan)) for r in rows], dtype=float)
            m = (t_idx >= 0) & (t_idx < T) & np.isfinite(aoi)
            u[t_idx[m]] = aoi[m]

        # Update raw accumulators
        finite_mask = np.isfinite(u)
        sum_raw[finite_mask] += u[finite_mask]
        cnt_raw[finite_mask] += 1.0

        # Compute user moving average (NaN-aware) and update accumulators
        mavg = moving_average_nan(u, mavg_window)
        finite_m = np.isfinite(mavg)
        sum_mavg[finite_m] += mavg[finite_m]
        cnt_mavg[finite_m] += 1.0

        del u, mavg

    # Final curves
    with np.errstate(invalid="ignore", divide="ignore"):
        sys_avg = sum_mavg / cnt_mavg
        raw_sys = sum_raw / cnt_raw

    # Episode-end markers / variance of RAW system average within each episode
    variances = []
    end_indices = []
    for ep in range(num_episodes):
        s = ep * episode_len
        e = s + episode_len
        variances.append(float(np.nanvar(raw_sys[s:e])))
        end_indices.append(e - 1)

    _ensure_dir(out_path)
    plt.figure()
    # Optionally decimate display
    if decimate > 1:
        xs = np.arange(T)[::decimate]
        plt.plot(xs, sys_avg[::decimate], linewidth=0.9, label=f"System avg (mean of user MAvgs, w={mavg_window})")
    else:
        plt.plot(sys_avg, linewidth=0.9, label=f"System avg (mean of user MAvgs, w={mavg_window})")

    for ep in range(1, num_episodes):
        x = ep * episode_len - 0.5
        plt.axvline(x, linewidth=0.5, linestyle="--", alpha=0.5)

    start_ep = 1 if skip_first_episode_var and num_episodes > 0 else 0
    if start_ep < num_episodes:
        xs_mark = end_indices[start_ep:]
        ys_mark = [sys_avg[i] for i in xs_mark]
        plt.scatter(xs_mark, ys_mark, s=12, marker="o", label="Episode end")

    plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)
    plt.xlabel("Global Slot Index")
    plt.ylabel("AoI")
    if title is None:
        title = "System Average AoI (Mean of Users' Moving Averages) with Episode Variance"
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_mavg_aoi_per_user_grid(
    data, num_slots, frames_per_episode, num_episodes,
    out_dir, out_pdf="All_Users_MovingAvgAoI.pdf",
    ncols=3,
    episode_tick=1,          # label x-ticks every k episodes
    y_min=300.0,             # start y-axis at 300
    annotate_variance=True,  # write variance text at episode end
    skip_first_variance=True,
    mavg_window=None         # None = cumulative avg (your example); int -> rolling window
):
    """
    Grid of per-user plots.
    - data: { uid: [ {'ep':1..E,'frame':0..F-1,'slot':0..S-1,'aoi':float}, ... ], ... }
    - Cumulative average by default (matches your example code).
      Set mavg_window (e.g., 100) to plot a rolling moving average instead.
    - Green hollow circle at EVERY episode end; variance annotated at episode end
      (skips ep=1 if skip_first_variance=True).
    """
    import os, math
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    # Helper: rolling mean without huge memory
    def rolling_mean(x, w):
        if w is None or w <= 1:
            return None  # signal to use cumulative average
        x = np.asarray(x, dtype=float)
        c = np.cumsum(np.insert(x, 0, 0.0))
        out = (c[w:] - c[:-w]) / float(w)
        # left-pad so length matches x
        pad = np.full(w - 1, out[0] if out.size > 0 else np.nan, dtype=float)
        return np.concatenate([pad, out])

    uids = sorted(data.keys(), key=lambda z: (isinstance(z, str), z))
    nrows = max(1, math.ceil(len(uids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.1 * nrows), squeeze=False)
    axes = axes.ravel()

    T_ep = num_slots * frames_per_episode
    T = T_ep * num_episodes
    # indices (global slots) where each episode ends
    episode_end_slots = [(e * T_ep) - 1 for e in range(1, num_episodes + 1)]
    # choose which episode ends appear as xticks
    xtick_slots = episode_end_slots[::max(1, episode_tick)]

    for ii, uid in enumerate(uids):
        ax = axes[ii]
        rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        if not rows:
            ax.set_title(f"U{uid} (no data)", fontsize=8)
            ax.axis("off")
            continue

        # Build dense vectors for this user (global time index & AoI values)
        t = []
        aoi_vals = []
        for r in rows:
            ep = int(r["ep"])
            fr = int(r["frame"])
            sl = int(r["slot"])
            g  = (ep - 1) * T_ep + fr * num_slots + sl
            if 0 <= g < T:
                t.append(g)
                aoi_vals.append(float(r.get("aoi", 0.0)))

        if not t:
            ax.set_title(f"U{uid} (empty)", fontsize=8)
            ax.axis("off")
            continue

        # Sort by time index
        order = np.argsort(t)
        t = np.asarray(t, dtype=int)[order]
        aoi_vals = np.asarray(aoi_vals, dtype=float)[order]

        # Compute average curve
        if mavg_window is None:
            # cumulative average (matches your example)
            csum = np.cumsum(aoi_vals)
            avg_curve = csum / (np.arange(aoi_vals.size, dtype=float) + 1.0)
        else:
            # rolling window average
            avg_curve = rolling_mean(aoi_vals, mavg_window)

        # Plot the average curve
        ax.plot(t, avg_curve, linestyle="-", linewidth=1.1, label=f"U{uid}")

        # Episode-end green hollow markers (on the curve value at that slot, if present)
        end_mask = np.isin(t, episode_end_slots)
        if np.any(end_mask):
            ax.plot(
                t[end_mask], avg_curve[end_mask],
                "o", markerfacecolor="none", markeredgecolor="green", markersize=5, linestyle="None"
            )

        # Variance per episode (on RAW AoI within each episode), annotate at episode end
        if annotate_variance:
            for ep in range(1, num_episodes + 1):
                if skip_first_variance and ep == 1:
                    continue
                s = (ep - 1) * T_ep
                e = ep * T_ep
                # pick raw AoI values within this episode (by global time)
                sel = (t >= s) & (t < e)
                if not np.any(sel):
                    continue
                v = float(np.var(aoi_vals[sel]))
                end_slot = e - 1
                # if we have the curve value at that exact slot, annotate there;
                # otherwise annotate at the closest available slot in that episode
                if end_slot in t:
                    yy = avg_curve[t.tolist().index(end_slot)]
                    xx = end_slot
                else:
                    # nearest index inside the episode
                    idx = np.where(sel)[0][-1]
                    xx = int(t[idx])
                    yy = float(avg_curve[idx])
                ax.annotate(f"var={v:.0f}", (xx, yy), xytext=(3, 3),
                            textcoords="offset points", fontsize=6)

        # Cosmetics
        ax.set_ylim(bottom=y_min)
        ax.set_xlim([0, T])
        ax.set_xticks(xtick_slots)
        ax.set_title(f"U{uid}", fontsize=8)
        ax.set_xlabel("Slot Index")
        ax.set_ylabel("Average AoI")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False, fontsize=7)

    # Hide any leftover empty axes
    for j in range(len(uids), nrows * ncols):
        axes[j].axis("off")

    fig.tight_layout(pad=0.6)
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved All Users Moving Avg AoI → {out_path}")

def plot_aoi_cdf_raw(
    data, num_slots, frames_per_episode, num_episodes,
    out_dir, out_pdf="AOI_Empirical_CDF_Raw.pdf",
    aoi_clip_max=None, show_ccdf=False, show_percentiles=(50,90,95,99),
):
    """
    ECDF over ALL raw per-slot AoI values across users and episodes.
    Uses only (ep, frame, slot) -> no _t required.
    """
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix",
        "axes.unicode_minus": False, "pdf.use14corefonts": True,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    T_ep = num_slots * frames_per_episode
    vals = []

    for uid in sorted(data.keys()):
        rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        for r in rows:
            ep = int(r["ep"]); fr = int(r["frame"]); sl = int(r["slot"])
            if not (1 <= ep <= num_episodes):
                continue
            # (We don’t actually need the global index; we just need AOI.)
            a = r.get("aoi", None)
            if a is None:
                continue
            try:
                a = float(a)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(a):
                continue
            if aoi_clip_max is not None:
                a = min(a, float(aoi_clip_max))
            vals.append(a)

    if not vals:
        print("[PLOT] No AoI to plot CDF.")
        return

    x = np.sort(np.array(vals, dtype=float))
    n = x.size
    y = np.arange(1, n + 1, dtype=float) / n

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    ax.set_title("Empirical CDF of Raw AoI (All Users & Slots)")
    ax.set_xlabel("AoI"); ax.set_ylabel("F(x) = P{AoI ≤ x}")
    ax.grid(True, alpha=0.3)
    ax.step(x, y, where="post", linewidth=1.3)

    if show_ccdf:
        ax.step(x, 1.0 - y, where="post", linestyle="--", linewidth=1.0, label="1 - F(x)")

    if show_percentiles:
        for p in show_percentiles:
            if 0 < p < 100:
                xp = np.percentile(x, p)
                ax.axvline(xp, linewidth=0.8, alpha=0.4)
                ax.text(xp, 0.02, f"P{p}≈{xp:.1f}", rotation=90, va="bottom", ha="right", fontsize=7)

    if show_ccdf:
        ax.legend(frameon=False, fontsize=8)

    out_path = os.path.join(out_dir, out_pdf)
    fig.tight_layout(); fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved raw AoI CDF → {out_path}")

def plot_system_aoi_cdf_from_ep_frame_slot(
    data, num_slots, frames_per_episode, num_episodes,
    out_dir, out_pdf="System_AOI_Empirical_CDF.pdf",
    mavg_window=100, aoi_clip_max=None,
    show_ccdf=False, show_percentiles=(50,90,95,99),
):
    """
    ECDF of the *system moving-average AoI*.
    Steps:
      1) Build a dense per-user vector u[g] from (ep,frame,slot) with AOI (NaN for missing).
      2) Rolling moving average per user (window=mavg_window), NaN-aware.
      3) System curve = slot-wise mean over users (where finite).
      4) ECDF over the system curve values.
    """
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix",
        "axes.unicode_minus": False, "pdf.use14corefonts": True,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    })

    def moving_average_nan(x: np.ndarray, w: int) -> np.ndarray:
        if w is None or w <= 1: return x.astype(float, copy=True)
        x = x.astype(float, copy=False)
        mask = np.isfinite(x).astype(float)
        x0 = np.where(np.isfinite(x), x, 0.0)
        kern = np.ones(w, dtype=float)
        num = np.convolve(x0,  kern, mode="full")[:x.size]
        den = np.convolve(mask, kern, mode="full")[:x.size]
        out = np.divide(num, den, out=np.full_like(num, np.nan), where=den>0)
        return out

    T_ep = num_slots * frames_per_episode
    T = T_ep * num_episodes

    sum_mavg = np.zeros(T, dtype=float)
    cnt_mavg = np.zeros(T, dtype=float)

    for uid in sorted(data.keys()):
        rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        u = np.full(T, np.nan, dtype=float)

        for r in rows:
            ep = int(r["ep"]); fr = int(r["frame"]); sl = int(r["slot"])
            if not (1 <= ep <= num_episodes):
                continue
            g = (ep - 1) * T_ep + fr * num_slots + sl
            if 0 <= g < T:
                a = r.get("aoi", None)
                if a is None:
                    continue
                try:
                    a = float(a)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(a):
                    continue
                if aoi_clip_max is not None:
                    a = min(a, float(aoi_clip_max))
                u[g] = a

        mavg = moving_average_nan(u, mavg_window)
        finite = np.isfinite(mavg)
        sum_mavg[finite] += mavg[finite]
        cnt_mavg[finite] += 1.0
        del u, mavg

    with np.errstate(invalid="ignore", divide="ignore"):
        sys_mavg = sum_mavg / cnt_mavg

    sys_mavg = sys_mavg[np.isfinite(sys_mavg)]
    if sys_mavg.size == 0:
        print("[PLOT] No finite system AoI values for CDF (check data fields or reduce mavg_window).")
        return

    x = np.sort(sys_mavg)
    n = x.size
    y = np.arange(1, n + 1, dtype=float) / n

    fig = plt.figure(figsize=(6.5, 4.0))
    ax = fig.add_subplot(111)
    ax.set_title("Empirical CDF of System Moving-Average AoI")
    ax.set_xlabel("System AoI"); ax.set_ylabel("F(x) = P{System AoI ≤ x}")
    ax.grid(True, alpha=0.3)
    ax.step(x, y, where="post", linewidth=1.3)

    if show_ccdf:
        ax.step(x, 1.0 - y, where="post", linestyle="--", linewidth=1.0, label="1 - F(x)")

    if show_percentiles:
        for p in show_percentiles:
            if 0 < p < 100:
                xp = np.percentile(x, p)
                ax.axvline(xp, linewidth=0.8, alpha=0.4)
                ax.text(xp, 0.02, f"P{p}≈{xp:.1f}", rotation=90, va="bottom", ha="right", fontsize=7)

    if show_ccdf:
        ax.legend(frameon=False, fontsize=8)

    out_path = os.path.join(out_dir, out_pdf)
    fig.tight_layout(); fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved system AoI CDF → {out_path}")


def make_run_dir(M_total, num_slots, num_episodes, gamma_th):
    #stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"AoI_U{M_total}_S{num_slots}_EP{num_episodes}_GTH{gamma_th}"
    out = os.path.join(name)
    os.makedirs(out, exist_ok=True)
    return out


# ------------------- Environment params (yours) -------------------
num_slots          = 12
frames_per_episode = 200
num_episodes       = 251
M_total            = 30

gamma_th_db        = -10
gamma_th           = 10 ** (gamma_th_db / 10.0)

# if you used a manual seed somewhere, define it here once
seed = 42
seed_value = globals().get("seed", None)


run_meta = {
    "M_total": int(M_total),
    "num_slots": int(num_slots),
    "num_episodes": int(num_episodes),
    "frames_per_episode": int(frames_per_episode) if "num_frames" in globals() else None,
    "seed": int(seed) if seed_value is not None else None,
    "torch_seed": int(torch.initial_seed()) if torch else None,
    "np_seed_state": str(np.random.get_state()[1][0]),
    "notes": "PPO AoI run"
}

RUN_DIR = make_run_dir(M_total, num_slots, num_episodes, gamma_th)


# 4. >>> SAVE META FILE HERE <<<
#with open(os.path.join(RUN_DIR, "meta.json"), "w") as f:
 #   json.dump(run_meta, f, indent=2)

print(f"[SAVE] Run dir accessed: {RUN_DIR}")

episode_len = _episode_slot_count(num_slots, frames_per_episode)

data = load_episode_telemetry(RUN_DIR, filename=f"slotwise_dataU{M_total}S{num_slots}.npy")

plot_dir = os.path.join(RUN_DIR, "CDF_and_Variance")
os.makedirs(plot_dir, exist_ok=True)   # ensures the directory exists


#plot_system_avg_aoi_with_episode_markers_and_variance(
#    data, num_slots, frames_per_episode, num_episodes,
#    out_dir=plot_dir,
#    rolling_window=100,                      # smooth helper curve
#    use_stddev_instead_of_variance=False       # if you prefer std-dev bars
#)






# Assuming you've already loaded your telemetry dict:
# data = load_episode_telemetry(RUN_DIR, filename=f"slotwise_dataU{M_total}S{num_slots}.npy")

plot_mavg_aoi_per_user_grid(
    data=data,
    num_slots=num_slots,
    frames_per_episode=frames_per_episode,
    num_episodes=num_episodes,
    out_dir=plot_dir,
    out_pdf="All_Users_MovingAvgAoI.pdf",
    ncols=3,                # 3 columns; rows auto
    episode_tick=5,         # label every 5th episode on x-axis (keeps ticks readable)
    y_min=300.0,            # y-axis starts at 300
    annotate_variance=True, # write var at episode ends
    skip_first_variance=True,
    mavg_window=None        # None=cumulative avg (match your example). Or set 100 for rolling.
)

plot_aoi_cdf_raw(
    data=data,
    num_slots=num_slots, frames_per_episode=frames_per_episode, num_episodes=num_episodes,
    out_dir=plot_dir,
    out_pdf="AOI_Empirical_CDF_Raw.pdf",
    aoi_clip_max=4000,       # cap, or None
    show_ccdf=True,
    show_percentiles=(50,90,95,99)
)


plot_system_aoi_cdf_from_ep_frame_slot(
    data=data,
    num_slots=num_slots, frames_per_episode=frames_per_episode, num_episodes=num_episodes,
    out_dir=plot_dir,
    out_pdf="System_AOI_Empirical_CDF.pdf",
    mavg_window=100,      # try 20–50 if logs are sparse
    aoi_clip_max=4000,    # cap outliers if needed
    show_ccdf=True,
    show_percentiles=(50,90,95,99)
)
