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





def plot_episode_reward_curves_from_sar(
    sar_log_dir,
    M_total,
    num_slots,
    frames_per_episode,
    out_dir=None,
    normalize_mode="global",   # "global" | "running"
    ma_window=10,
    fig3_mode="cumulative",    # "cumulative" | "per-episode"
    # --- convergence extras ---
    roll_window=10,            # rolling window (episodes) for mean/std/slope
    ewma_alpha=0.15,           # 0<alpha<=1 for EWMA; lower=more smoothing
    stability_tail=10          # show stability stats over last K episodes
):
    """
    Adds convergence diagnostics:
      (d) Cumulative average + shrinking ±1σ band
      (e) Rolling mean ± std band (window=roll_window)
      (f) EWMA smooth of per-episode average
      (g) Rolling slope (finite-diff) of rolling mean
      (h) Stability panel over last K episodes (histogram + text)
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # === IEEE two-column figure setup ===
    IEEE_WIDTH = 3.4  # inches per column
    IEEE_HEIGHT = 2.1

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        # --- Larger readable labels ---
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.8,
        "axes.titlesize": 9,
        # --- Lines and grid aesthetics ---
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    import os, pickle, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

    # ---------- paths ----------
    if out_dir is None:
        out_dir = sar_log_dir
    os.makedirs(out_dir, exist_ok=True)
    sar_log_path = os.path.join(sar_log_dir, f"sar_logU{M_total}S{num_slots}.pkl")
    if not os.path.exists(sar_log_path):
        raise FileNotFoundError(f"Missing SAR log: {sar_log_path}")

    # ---------- load & canonicalize ----------
    with open(sar_log_path, "rb") as f:
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
            raise TypeError(f"Unsupported SAR format: {type(sar)}")
        records = [(-1, -1, i, rewards[i]) for i in range(len(rewards))]

    records.sort(key=lambda t: (t[0], t[1], t[2]))

    none_idx = [i for i, rec in enumerate(records) if rec[3] is None]
    if none_idx:
        raise ValueError(f"{len(none_idx)} SAR entries have reward=None. Fill/update rewards before plotting.")

    rewards = np.asarray([float(rec[3]) for rec in records], dtype=float)

    # ---------- episodes ----------
    slots_per_episode = int(num_slots) * int(frames_per_episode)
    if slots_per_episode <= 0:
        raise ValueError("slots_per_episode must be positive.")
    total_slots = len(rewards)
    n_episodes = total_slots // slots_per_episode
    remainder = total_slots % slots_per_episode
    if n_episodes == 0:
        raise ValueError(f"Not enough slots ({total_slots}) for one episode of {slots_per_episode} slots.")
    if remainder:
        rewards = rewards[: n_episodes * slots_per_episode]

    R = rewards.reshape(n_episodes, slots_per_episode)

    # ---------- base metrics ----------
    ep_sum = R.sum(axis=1)          # sum per episode
    ep_avg = R.mean(axis=1)         # avg per-slot per episode
    ep_idx = np.arange(1, n_episodes + 1)

    # ---------- your Fig-3 base ----------
    def minmax_norm(y):
        lo, hi = np.min(y), np.max(y)
        return (y - lo) / max(hi - lo, 1e-8)

    def running_norm(y):
        out = np.zeros_like(y, dtype=float)
        run_min, run_max = np.inf, -np.inf
        for i, v in enumerate(y):
            run_min = v if v < run_min else run_min
            run_max = v if v > run_max else run_max
            out[i] = (v - run_min) / max(run_max - run_min, 1e-8)
        return out

    if fig3_mode == "cumulative":
        base = np.cumsum(ep_avg) / np.arange(1, n_episodes + 1)  # cumulative average of ep_avg
        ylabel = "Normalized Avg CUMULATIVE Reward [0–1]"
        title_c = "Fig-3 Style Normalized Cumulative Reward"
    elif fig3_mode == "per-episode":
        base = ep_avg.copy()
        ylabel = "Normalized Avg Reward [0–1]"
        title_c = "Fig-3 Style Normalized Per-Episode Reward"
    else:
        raise ValueError("fig3_mode must be 'cumulative' or 'per-episode'.")

    ep_norm = running_norm(base) if normalize_mode == "running" else minmax_norm(base)

    # ---------- helpers ----------
    def moving_avg(y, k):
        k = int(max(1, k))
        if k == 1 or len(y) < k:
            return y, np.arange(len(y))
        ma = np.convolve(y, np.ones(k)/k, mode="valid")
        x = np.arange(k-1, k-1+len(ma))
        return ma, x

    def rolling_std(y, k):
        k = int(max(1, k))
        if k <= 1 or len(y) < k:
            return np.full_like(y, np.nan, dtype=float)
        out = np.full_like(y, np.nan, dtype=float)
        for i in range(k-1, len(y)):
            seg = y[i-k+1:i+1]
            out[i] = np.std(seg)
        return out

    def rolling_slope(y, k):
        # slope of rolling mean (finite difference)
        m, _ = moving_avg(y, k)
        # align back to episode indices used in moving_avg
        x = np.arange(len(y))
        xs = np.arange(k-1, k-1+len(m))
        dy = np.diff(m, prepend=m[0])
        slope = np.full_like(y, np.nan, dtype=float)
        slope[xs] = dy
        return slope

    # ---------- (a) Raw sum ----------
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.plot(ep_idx, ep_sum, alpha=0.35, label="Episode Sum (raw)")
    if ma_window and n_episodes >= ma_window:
        ma, x = moving_avg(ep_sum, ma_window)
        plt.plot(x+1, ma, linewidth=1.6, label=f"Moving Avg (k={ma_window})")
    plt.xlabel("Episode"); plt.ylabel("Sum of Reward")
    plt.title("Per-Episode Raw Sum of Reward")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    out_a = os.path.join(out_dir, f"episode_reward_sum_U{M_total}S{num_slots}.pdf")
    plt.savefig(out_a, dpi=600, bbox_inches="tight"); plt.close()

    # ---------- (b) Raw average ----------
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.plot(ep_idx, ep_avg, alpha=0.35, label="Episode Average (raw)")
    if ma_window and n_episodes >= ma_window:
        ma, x = moving_avg(ep_avg, ma_window)
        plt.plot(x+1, ma, linewidth=1.6, label=f"Moving Avg (k={ma_window})")
    plt.xlabel("Episode"); plt.ylabel("Average Reward per Slot")
    plt.title("Per-Episode Raw Average Reward")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    out_b = os.path.join(out_dir, f"episode_reward_avg_U{M_total}S{num_slots}.pdf")
    plt.savefig(out_b, dpi=600, bbox_inches="tight"); plt.close()

    # ---------- (c) Normalized Fig-3 style ----------
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.plot(ep_idx, ep_norm, alpha=0.35, label=f"Normalized ({normalize_mode}, {fig3_mode})")
    if ma_window and n_episodes >= ma_window:
        ma, x = moving_avg(ep_norm, ma_window)
        plt.plot(x+1, ma, linewidth=1.6, label=f"Moving Avg (k={ma_window})")
    plt.xlabel("Episode"); plt.ylabel(ylabel)
    plt.title(title_c)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    out_c = os.path.join(out_dir, f"episode_reward_avg_normalized_{normalize_mode}_{fig3_mode}_U{M_total}S{num_slots}.pdf")
    plt.savefig(out_c, dpi=600, bbox_inches="tight"); plt.close()

    # ---------- (d) CUMULATIVE average with ±1σ band ----------
    cum_mean = np.cumsum(ep_avg) / np.arange(1, n_episodes+1)
    # cumulative std of ep_avg sequence (online)
    cstd = np.zeros(n_episodes, dtype=float)
    mean_run = 0.0
    M2 = 0.0
    for i, v in enumerate(ep_avg, start=1):
        delta = v - mean_run
        mean_run += delta / i
        M2 += delta * (v - mean_run)
        cstd[i-1] = np.sqrt(M2 / i) if i > 0 else 0.0

    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.plot(ep_idx, cum_mean, linewidth=1.4, label="Cumulative mean of ep-avg")
    upper, lower = cum_mean + cstd, cum_mean - cstd
    plt.fill_between(ep_idx, lower, upper, alpha=0.18, label="±1σ band (cumulative)")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    #plt.title("Convergence: Cumulative Mean ±1σ")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    out_d = os.path.join(out_dir, f"reward_convergence_cummean_band_U{M_total}S{num_slots}.pdf")
    plt.savefig(out_d, dpi=600, bbox_inches="tight"); plt.close()

    # ---------- (e) Rolling mean ± std band ----------
    k = max(2, roll_window)
    rmean, rx = moving_avg(ep_avg, k)  # length N-k+1, aligned to indices (k-1 .. N-1)
    rstd = rolling_std(ep_avg, k)  # length N, NaN for 0..k-2, valid for k-1..N-1

    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.plot(ep_idx, ep_avg, alpha=0.20, label="Episode avg (raw)")

    # Build a full-length array for rmean aligned to episode indices
    rmean_full = np.full_like(ep_avg, np.nan, dtype=float)
    if len(rmean) > 0:
        start = k - 1  # first index where rolling stats are valid
        rmean_full[start:start + len(rmean)] = rmean

    # Mask to positions where both rmean and rstd are defined
    mask = np.isfinite(rmean_full) & np.isfinite(rstd)
    x = ep_idx[mask]
    mu = rmean_full[mask]
    sd = rstd[mask]

    # Plot aligned rolling mean and its ±1σ band
    if x.size > 0:
        plt.plot(x, mu, linewidth=1.6, label=f"Rolling mean (k={k})")
        plt.fill_between(x, mu - sd, mu + sd, alpha=0.12, label="±1σ (rolling)")

    plt.xlabel("Episode");
    plt.ylabel("Average Reward per Slot")
    plt.title("Convergence: Rolling Mean ±1σ")
    plt.grid(True, alpha=0.3);
    plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    out_e = os.path.join(out_dir, f"reward_convergence_rolling_band_U{M_total}S{num_slots}.pdf")
    plt.savefig(out_e, dpi=600, bbox_inches="tight");
    plt.close()

    # ---------- (f) EWMA of ep_avg ----------
    ewma = np.zeros_like(ep_avg, dtype=float)
    if len(ep_avg):
        ewma[0] = ep_avg[0]
        for i in range(1, len(ep_avg)):
            ewma[i] = ewma_alpha * ep_avg[i] + (1 - ewma_alpha) * ewma[i-1]
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.plot(ep_idx, ep_avg, alpha=0.25, label="Episode avg (raw)")
    plt.plot(ep_idx, ewma, linewidth=1.6, label=f"EWMA (alpha={ewma_alpha})")
    plt.xlabel("Episode"); plt.ylabel("Average Reward per Slot")
    plt.title("Convergence: EWMA Trend")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    out_f = os.path.join(out_dir, f"reward_convergence_ewma_U{M_total}S{num_slots}.pdf")
    plt.savefig(out_f, dpi=600, bbox_inches="tight"); plt.close()

    # ---------- (g) Rolling slope of the rolling mean ----------
    rslope = rolling_slope(ep_avg, max(2, roll_window))
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.plot(ep_idx, rslope, linewidth=1.3, label=f"Δ(rolling mean), k={roll_window}")
    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("Episode"); plt.ylabel("Δ Reward")
    plt.title("Convergence: Rolling Mean Slope (→ 0 when converged)")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    out_g = os.path.join(out_dir, f"reward_convergence_slope_U{M_total}S{num_slots}.pdf")
    plt.savefig(out_g, dpi=600, bbox_inches="tight"); plt.close()

    # ---------- (h) Stability tail panel (last K episodes) ----------
    K = int(max(1, min(stability_tail, n_episodes)))
    tail = ep_avg[-K:] if K <= len(ep_avg) else ep_avg
    t_mu, t_sd = float(np.mean(tail)), float(np.std(tail)) if len(tail) > 1 else 0.0
    plt.figure(figsize=(6.6, 3.0))
    plt.hist(tail, bins=min(15, max(5, K//2)), alpha=0.8)
    plt.xlabel("Average Reward per Slot (last K episodes)")
    plt.ylabel("Count")
    plt.title(f"Stability (last {K} episodes): mean={t_mu:.3f}, std={t_sd:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_h = os.path.join(out_dir, f"reward_convergence_tail_hist_U{M_total}S{num_slots}.pdf")
    plt.savefig(out_h, dpi=600, bbox_inches="tight"); plt.close()

    print(
        "[PLOT] Saved:\n"
        f"  {out_a}\n  {out_b}\n  {out_c}\n  {out_d}\n  {out_e}\n  {out_f}\n  {out_g}\n  {out_h}"
    )

    return {
        "episode_index": ep_idx,
        "episode_sum": ep_sum,
        "episode_avg": ep_avg,
        "normalized_curve": ep_norm,
        "cum_mean": cum_mean,
        "cum_std": cstd,
        "rolling_mean": rmean if 'rmean' in locals() else None,
        "rolling_std": rstd if 'rstd' in locals() else None,
        "ewma": ewma,
        "rolling_slope": rslope,
        "tail_mean_std": (t_mu, t_sd),
    }

def make_run_dir(M_total, num_slots, num_episodes, gamma_th, tau):
    #stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"AoI_U{M_total}_S{num_slots}_EP{num_episodes}_RewardNS"
    out = os.path.join(name)
    os.makedirs(out, exist_ok=True)
    return out



num_slots = 15
frames_per_episode = 200
num_episodes = 100
user_counts = [60, 65, 70, 75]

run_dirs = [f"AoI_U{u}_S{num_slots}_EP{num_episodes}_RewardNS" for u in user_counts]

# Get the scalar per run + plot
xs, ys = plot_episode_reward_curves_from_sar(run_dirs, user_counts, num_slots, frames_per_episode, num_episodes, out_pdf="Reward_vs_users.pdf")  # rename as you like



