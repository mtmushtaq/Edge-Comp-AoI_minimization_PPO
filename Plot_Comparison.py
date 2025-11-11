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

def compute_user_moving_avgs(data, num_slots, frames_per_episode, num_episodes):
    """
    For each user, compute the individual moving average exactly like your per-user plot:
      mov_avg_u[k] = (AoI_u[0] + ... + AoI_u[k]) / (k+1)
    Also returns each user's global time indices t_k = (ep-1)*T + frame*num_slots + slot.
    """
    import numpy as np

    T = num_slots * frames_per_episode
    user_series = {}   # uid -> dict with 't' and 'mavg'

    for uid in sorted(data.keys()):
        rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        t_list, mavg_list = [], []
        csum = 0.0
        for i, r in enumerate(rows):
            t = (r["ep"] - 1) * T + r["frame"] * num_slots + r["slot"]
            aoi = float(r.get("aoi", 0.0))
            csum += aoi
            mavg = csum / (i + 1)  # <-- exactly as specified
            t_list.append(t)
            mavg_list.append(mavg)
        user_series[uid] = {
            "t": np.array(t_list, dtype=int),
            "mavg": np.array(mavg_list, dtype=float),
        }
    return user_series

def plot_system_avg_aoi_timewise_strict(
    data_ppo,
    data_random,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir,
    out_pdf="Comparison_system_aoi_time_avg.pdf",
    also_plot_mean_of_user_mavgs=True,
    rolling_window=None,  # e.g., 1000 for a smoother auxiliary curve
    # --- new knobs ---
    include_per_slot_in_main=True,  # True -> keep thin per-slot mean in the main figure
    save_avg_only=True,  # True -> also save a second "averages-only" figure
    avg_only_pdf="system_aoi_time_avg_only.pdf",
    avg_ylim_clip=(1, 99),  # y-axis clips by percentiles of average curves for better scale
):
    """
    Builds:
      - per-slot mean AoI across users at each global t (O(N))
      - running mean over time (AAoI-like system convergence)
      - optional: mean of users' moving averages at each t (aligned & forward-filled)

    Saves:
      - main plot (optionally includes per-slot mean)
      - averages-only plot (running mean, rolling mean, mean of user mavgs)
    """
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

    # ------- styling -------
    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix", "axes.unicode_minus": False, "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    uids_ppo = sorted(data_ppo.keys())
    uids_random = sorted(data_random.keys())

    U = len(uids_ppo)  # Assuming same users for PPO and Random
    T = num_slots * frames_per_episode
    T_total = T * num_episodes

    # --- A) Per-slot mean AoI across users (cross-sectional) ---
    def compute_per_slot_mean(data):
        sum_per_t, cnt_per_t = {}, {}
        for uid in data:
            rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
            for r in rows:
                t = (r["ep"] - 1) * T + r["frame"] * num_slots + r["slot"]
                aoi = float(r.get("aoi", 0.0))
                sum_per_t[t] = sum_per_t.get(t, 0.0) + aoi
                cnt_per_t[t] = cnt_per_t.get(t, 0) + 1

        if not sum_per_t:
            return np.nan, None

        t_sorted = np.array(sorted(sum_per_t.keys()), dtype=int)
        max_t = int(min(t_sorted.max(), T_total))
        per_slot_mean = np.full(max_t + 1, np.nan)
        counts = np.zeros(max_t + 1, dtype=int)
        for t in t_sorted:
            if t <= max_t:
                per_slot_mean[t] = sum_per_t[t] / max(cnt_per_t[t], 1)
                counts[t] = cnt_per_t[t]

        # forward-fill gaps so cumulative is stable (optional but helpful)
        mask = np.isfinite(per_slot_mean)
        if mask.any():
            last = np.nan
            for i in range(per_slot_mean.size):
                if np.isfinite(per_slot_mean[i]):
                    last = per_slot_mean[i]
                else:
                    per_slot_mean[i] = last

        return per_slot_mean, mask

    per_slot_mean_ppo, mask_ppo = compute_per_slot_mean(data_ppo)
    per_slot_mean_random, mask_random = compute_per_slot_mean(data_random)

    # Running (cumulative) mean over time (AAoI-like)
    def compute_running_mean(per_slot_mean):
        valid = np.isfinite(per_slot_mean)
        ps = np.where(valid, per_slot_mean, 0.0)
        w = np.where(valid, 1.0, 0.0)
        csum = np.cumsum(ps)
        wsum = np.cumsum(w)
        return np.divide(csum, np.maximum(wsum, 1e-12))

    system_running_mean_ppo = compute_running_mean(per_slot_mean_ppo)
    system_running_mean_random = compute_running_mean(per_slot_mean_random)

    # Optional rolling mean (centered simple moving average)
    roll_curve_ppo, roll_curve_random = None, None
    if rolling_window and rolling_window > 1:
        k = int(rolling_window)
        kernel = np.ones(k) / k
        num_ppo = np.convolve(per_slot_mean_ppo, kernel, mode="same")
        den_ppo = np.convolve(np.ones_like(per_slot_mean_ppo), kernel, mode="same")
        roll_curve_ppo = np.divide(num_ppo, np.maximum(den_ppo, 1e-12))

        num_random = np.convolve(per_slot_mean_random, kernel, mode="same")
        den_random = np.convolve(np.ones_like(per_slot_mean_random), kernel, mode="same")
        roll_curve_random = np.divide(num_random, np.maximum(den_random, 1e-12))

    # --- Plotting both PPO and Random on the same graph ---
    x = np.arange(len(system_running_mean_ppo))

    fig, ax = plt.subplots(figsize=(7.8, 3.2))

    le = min(len(system_running_mean_ppo), len(system_running_mean_random))
    ax.plot(x[0:le], system_running_mean_ppo[0:le], linewidth=1.6, label="PPO (Running mean AoI)", color="blue")
    ax.plot(x[0:le], system_running_mean_random[0:le], linewidth=1.6, label="Random (Running mean AoI)", color="red")

    if roll_curve_ppo is not None:
        ax.plot(x, roll_curve_ppo, linestyle="--", linewidth=1.2, label="PPO (Rolling mean)", color="blue", alpha=0.7)

    if roll_curve_random is not None:
        ax.plot(x, roll_curve_random, linestyle="--", linewidth=1.2, label="Random (Rolling mean)", color="red", alpha=0.7)

    ax.set_xlabel("Time (Slots)")
    ax.set_ylabel("Average AoI")
    ax.set_title("Comparison of System Average AoI (PPO vs Random)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Comparison System AoI plot saved → {out_path}")

    # Saving the averages-only plot
    if save_avg_only:
        fig2, ax2 = plt.subplots(figsize=(7.8, 3.2))
        ax2.plot(x[0:le], system_running_mean_ppo[0:le], linewidth=1.6, label="PPO (Running mean AoI)", color="blue")
        ax2.plot(x[0:le], system_running_mean_random[0:le], linewidth=1.6, label="Random (Running mean AoI)", color="red")
        ax2.set_xlabel("Time (Slots)")
        ax2.set_ylabel("Average AoI")
        ax2.set_title("Averages Only: System Average AoI (PPO vs Random)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig2.tight_layout()
        out_path2 = os.path.join(out_dir, avg_only_pdf)
        fig2.savefig(out_path2, dpi=600, format="pdf", bbox_inches="tight")
        plt.close(fig2)
        print(f"[PLOT] System AoI (averages-only) saved → {out_path2}")



def make_run_dir(M_total, num_slots, method):
    #stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if method == "SLT":

        name = f"AoI_U{M_total}_S{num_slots}_SLT"
        out = os.path.join(name)
        os.makedirs(out, exist_ok=True)
        return out
    elif method == "RNDM":
        name = f"AoI_U{M_total}_S{num_slots}_RNDM"
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


def plot_policy_analytics_modern_comparison(
        sar_log_dir_ppo,
        sar_log_dir_random,
        M_total,
        num_slots,
        out_dir=None,  # if None, saves alongside SAR log dir
        n_bins_profile=30,
        roll_smooth=3,
        jitter=0.08,
        hex_gridsize=35,
        dpi=150,
        # styling / behavior
        pretty_labels=True,
        xclip_quantiles=(0.5, 99.5),
        hex_min_percentile=60,
        hex_mincnt=3,
        make_policy_contours=True,
        contour_bins=60,
        contour_sigma=1.2,
):
    """
    Colorful policy analytics comparing PPO and Random:
      1) Policy profiles with bands: P(a | AoI_near), P(a | AoI_far)
      2) Hex-multiples (focused, log-scaled, de-noised)
      3) Smoothed 2D policy contours (frontiers)
    Saves in the SAME directory as the SAR log by default.
    """
    import os, pickle, math
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

    # ---------- Define missing variables and functions ----------
    BAND_ALPHA = 0.18
    ACCENT_DARK = "#4B4453"  # titles/lines; neutral-dark

    def _lbl_near():
        return r"AoI$_{\mathrm{near}}$"  # Near-user AoI

    def _lbl_far():
        return r"AoI$_{\mathrm{far}}$"  # Far-user AoI

    def _robust_limits(x):
        lo, hi = np.nanpercentile(x, [0.5, 99.5])
        if not np.isfinite(lo): lo = np.nanmin(x)
        if not np.isfinite(hi): hi = np.nanmax(x)
        if lo == hi: hi = lo + 1e-9
        return lo, hi

    def distinct_colors(K):
        BASE_COLORS = [
            "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
            "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
            "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
            "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
        ]
        if K <= len(BASE_COLORS):
            return BASE_COLORS[:K]
        cols = []
        i = 0
        while len(cols) < K:
            cols.append(BASE_COLORS[i % len(BASE_COLORS)])
            i += 1
        return cols

    # ---------- Paths ----------
    if out_dir is None:
        out_dir = sar_log_dir_ppo
    os.makedirs(out_dir, exist_ok=True)

    # Load PPO SAR log
    sar_log_path_ppo = os.path.join(sar_log_dir_ppo, f"sar_logU{M_total}S{num_slots}.pkl")
    if not os.path.exists(sar_log_path_ppo):
        raise FileNotFoundError(f"Missing PPO SAR log: {sar_log_path_ppo}")

    # Load Random SAR log
    sar_log_path_random = os.path.join(sar_log_dir_random, f"sar_logU{M_total}S{num_slots}.pkl")
    if not os.path.exists(sar_log_path_random):
        raise FileNotFoundError(f"Missing Random SAR log: {sar_log_path_random}")

    # ---------- Load SAR logs ----------
    def load_sar_log(sar_log_path):
        with open(sar_log_path, "rb") as f:
            sar = pickle.load(f)
        states, actions = [], []

        def _push(s, a):
            states.append(np.asarray(s)); actions.append(int(a))

        if isinstance(sar, dict):
            s_key = next(k for k in sar if k.lower() in ["s", "state", "states"])
            a_key = next(k for k in sar if k.lower() in ["a", "action", "actions"])
            S, A = sar[s_key], sar[a_key]
            for i in range(min(len(S), len(A))): _push(S[i], A[i])
        elif isinstance(sar, (list, tuple)):
            for item in sar:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    _push(item[0], item[1])
                elif isinstance(item, dict):
                    s = item.get("s", item.get("state"));
                    a = item.get("a", item.get("action"))
                    if s is not None and a is not None: _push(s, a)
        else:
            raise TypeError("Unsupported SAR format")

        S = np.vstack(states)
        A = np.array(actions, int)
        if S.ndim != 2 or S.shape[1] < 2:
            raise ValueError(f"State shape {S.shape} — need at least 2 dims (AoI_near, AoI_far).")

        AoI_n = S[:, 0].astype(float)  # near
        AoI_f = S[:, 1].astype(float)  # far
        mask = np.isfinite(AoI_n) & np.isfinite(AoI_f) & np.isfinite(A)
        AoI_n, AoI_f, A = AoI_n[mask], AoI_f[mask], A[mask]
        return AoI_n, AoI_f, A

    # Load PPO and Random SAR data
    AoI_n_ppo, AoI_f_ppo, A_ppo = load_sar_log(sar_log_path_ppo)
    AoI_n_random, AoI_f_random, A_random = load_sar_log(sar_log_path_random)

    actions_unique = np.sort(np.unique(A_ppo))  # Assuming same actions for PPO and Random
    K = len(actions_unique)
    COLORS = distinct_colors(K)
    act2idx = {a: i for i, a in enumerate(actions_unique)}

    def col_of(a):
        return COLORS[act2idx[a]]

    # ---------- Policy profiles ----------
    def _profile(x, a, n_bins):
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(x, qs)
        edges = np.unique(edges)
        if len(edges) - 1 < max(5, n_bins // 2):
            edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        p = np.zeros(len(centers))
        n = np.zeros(len(centers), int)
        for i in range(len(centers)):
            m = (x >= edges[i]) & (x < edges[i + 1]) if i < len(centers) - 1 else (x >= edges[i]) & (x <= edges[i + 1])
            n[i] = m.sum()
            if n[i] > 0:
                p[i] = np.mean(A[m] == a)  # Fix applied here using np.mean()

        if roll_smooth > 1 and len(centers) >= roll_smooth:
            w = roll_smooth
            pad = w // 2

            def _ma(v):
                vv = np.pad(v, (pad, pad), mode="edge")
                out = np.convolve(vv, np.ones(w) / w, mode="valid")
                return out[:len(v)]

            p = _ma(p)
            n = np.maximum(_ma(n.astype(float)), 1e-12)

        se = np.sqrt(np.maximum(p * (1 - p) / np.maximum(n, 1), 1e-12))
        lo, hi = np.clip(p - 1.96 * se, 0, 1), np.clip(p + 1.96 * se, 0, 1)
        return centers, p, lo, hi

    # ---------- (1) Policy profiles ----------
    def _policy_profiles(x, x_label, tag):
        fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=dpi)
        for a in actions_unique:
            c, p, lo, hi = _profile(x, a, n_bins_profile)
            ax.plot(c, p, color=col_of(a), label=f"a={a}", linewidth=1.8)
            ax.fill_between(c, lo, hi, color=col_of(a), alpha=BAND_ALPHA, linewidth=0)
        ax.set_xlabel(x_label);
        ax.set_ylabel(r"$P(\mathrm{action}\mid \mathrm{AoI})$")
        ax.set_title(f"Policy Profiles vs {x_label}", color=ACCENT_DARK)
        ax.grid(True, alpha=0.35)
        ax.legend(title="Actions", ncols=min(K, 4), fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"policy_profiles_comparison_{tag}_U{M_total}S{num_slots}.pdf"),
                    format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    _policy_profiles(AoI_n_ppo, _lbl_near(), "PPO_AoIn")
    _policy_profiles(AoI_f_ppo, _lbl_far(), "PPO_AoIf")
    _policy_profiles(AoI_n_random, _lbl_near(), "Random_AoIn")
    _policy_profiles(AoI_f_random, _lbl_far(), "Random_AoIf")

    # ---------- (2) Hex-multiples (focused, denoised) ----------
    ncols = min(3, K);
    nrows = math.ceil(K / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), dpi=dpi, squeeze=False)
    for i, a in enumerate(actions_unique):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        m_ppo = (A_ppo == a)
        m_random = (A_random == a)
        if m_ppo.sum() > 0:
            xv_ppo, yv_ppo = AoI_n_ppo[m_ppo], AoI_f_ppo[m_ppo]
            xlo, xhi = _robust_limits(xv_ppo)
            ylo, yhi = _robust_limits(yv_ppo)
            # colorful, perceptually-uniform colormap for density
            hb = ax.hexbin(
                xv_ppo, yv_ppo,
                gridsize=hex_gridsize,
                mincnt=max(1, int(hex_mincnt)),
                norm=LogNorm(),
                cmap="viridis",
            )
            counts = hb.get_array()
            if counts.size:
                thr = np.percentile(counts, hex_min_percentile)
                hb.set_clim(vmin=max(thr, 1))  # hide low-density bins
                cmap = hb.get_cmap().copy()
                cmap.set_under(alpha=0.0)
                hb.set_cmap(cmap)
            ax.set_xlim(xlo, xhi);
            ax.set_ylim(ylo, yhi)
        ax.set_title(f"Action a={a}", fontsize=8, color=ACCENT_DARK)
        ax.set_xlabel(_lbl_near());
        ax.set_ylabel(_lbl_far())
        ax.grid(True, alpha=0.2)
    for j in range(K, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle("Hex Density of (AoI_near, AoI_far) per Action (PPO vs Random)", y=0.995, fontsize=9,
                 color=ACCENT_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, f"hex_multiples_comparison_U{M_total}S{num_slots}.pdf"),
                format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ---------- (3) Smoothed 2D policy contours ----------
    if make_policy_contours and K >= 2:
        def _gaussian_kernel1d(sigma, radius):
            x = np.arange(-radius, radius + 1, dtype=float)
            k = np.exp(-(x * x) / (2 * sigma * sigma))
            k /= k.sum()
            return k

        def _blur2d(arr, sigma):
            if sigma <= 0: return arr
            radius = int(max(1, round(3 * sigma)))
            k = _gaussian_kernel1d(sigma, radius)
            tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, arr)
            out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, tmp)
            return out

        def _edges(v):
            lo, hi = _robust_limits(v)
            return np.linspace(lo, hi, contour_bins + 1)

        xe, ye = _edges(AoI_n_ppo), _edges(AoI_f_ppo)
        ix = np.clip(np.digitize(AoI_n_ppo, xe) - 1, 0, contour_bins - 1)
        iy = np.clip(np.digitize(AoI_f_ppo, ye) - 1, 0, contour_bins - 1)

        counts = np.zeros((K, contour_bins, contour_bins), dtype=float)
        for cx, cy, a in zip(ix, iy, A_ppo):
            counts[act2idx[a], cx, cy] += 1.0

        probs = np.zeros_like(counts)
        for k_idx in range(K):
            probs[k_idx] = _blur2d(counts[k_idx], contour_sigma)
        denom = np.maximum(probs.sum(axis=0), 1e-12)
        probs /= denom

        Xc = 0.5 * (xe[:-1] + xe[1:])
        Yc = 0.5 * (ye[:-1] + ye[1:])
        Xg, Yg = np.meshgrid(Xc, Yc, indexing="ij")

        # region coloring with action colors
        region = np.argmax(probs, axis=0)  # [X,Y] in {0..K-1}
        region_cmap = ListedColormap(COLORS)
        norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), K)

        fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=dpi)
        ax.contourf(Xg, Yg, region.T, levels=np.arange(-0.5, K + 0.5, 1),
                    cmap=region_cmap, norm=norm, alpha=0.35)

        # draw p=0.5 contours per action in that action's line color
        for k_idx, a in enumerate(actions_unique):
            try:
                ax.contour(Xg, Yg, probs[k_idx].T, levels=[0.5],
                           colors=[col_of(a)], linewidths=1.6)
            except Exception:
                pass

        ax.set_xlabel(_lbl_near());
        ax.set_ylabel(_lbl_far())
        ax.set_title("Smoothed Policy Frontiers (PPO vs Random)", color=ACCENT_DARK)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"policy_contours_comparison_U{M_total}S{num_slots}.pdf"),
                    format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    print("[PLOT] Saved to:", out_dir)


def plot_time_averaged_system_aoi_comparison(data_ppo, data_random, num_slots, frames_per_episode, out_dir, out_pdf="system_aoi_time_avg_comparison.pdf"):
    import os, numpy as np, matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.1, "grid.linewidth": 0.5,
    })
    os.makedirs(out_dir, exist_ok=True)

    def compute_time_averaged_aoi(data):
        uids = sorted(data.keys())
        num_episodes = max(r["ep"] for uid in uids for r in data[uid]) + 1
        system_avg = []

        for ep in range(1, num_episodes):
            total = 0.0
            count = 0
            for uid in uids:
                aoi_vals = [r["aoi"] for r in data[uid] if r["ep"] == ep]
                total += sum(aoi_vals)
                count += len(aoi_vals)
            avg = total / count if count > 0 else np.nan
            system_avg.append(avg)

        return system_avg

    # Compute time-averaged AoI for PPO and Random
    system_avg_ppo = compute_time_averaged_aoi(data_ppo)
    system_avg_random = compute_time_averaged_aoi(data_random)

    # Plot the comparison
    plt.figure(figsize=(4.5, 2.6))
    x = np.arange(1, min(len(system_avg_ppo), len(system_avg_random)))

    plt.plot(x, system_avg_ppo[0:len(x)], color="blue", marker="o", markerfacecolor="none",
             markeredgecolor="green", linestyle='-', label="PPO")
    plt.plot(x, system_avg_random[0:len(x)], color="red", marker="s", markerfacecolor="none",
             markeredgecolor="orange", linestyle='--', label="Random")

    plt.xlabel("Episode")
    plt.ylabel("System Avg AoI (Time-Averaged)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the comparison plot
    plt.savefig(os.path.join(out_dir, out_pdf), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved time-averaged system AoI comparison → {out_pdf}")


def plot_system_avg_aoi_timewise_comparison(data_ppo, data_random, num_slots, frames_per_episode, num_episodes, out_dir,
                                            out_pdf="system_aoi_time_avg_comparison.pdf"):
    """
    Plots the system average AoI for both PPO and Random policies.
    """
    # Compute AAoI for both PPO and Random
    system_avg_ppo = plot_system_avg_aoi_timewise_strict(data_ppo, num_slots, frames_per_episode, num_episodes, out_dir,
                                                         out_pdf="system_aoi_time_avg_ppo.pdf")
    system_avg_random = plot_system_avg_aoi_timewise_strict(data_random, num_slots, frames_per_episode, num_episodes,
                                                            out_dir, out_pdf="system_aoi_time_avg_random.pdf")

    # Comparison plot
    plt.figure(figsize=(7, 5))
    plt.plot(system_avg_ppo, label="PPO", color="blue", linestyle='--')
    plt.plot(system_avg_random, label="Random", color="red", linestyle='--')

    plt.title("System Average AoI Comparison (PPO vs Random)")
    plt.xlabel("Time (Episode)")
    plt.ylabel("System Average AoI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    comparison_pdf_path = os.path.join(out_dir, out_pdf)
    plt.savefig(comparison_pdf_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Comparison plot saved at {comparison_pdf_path}")



# 4. >>> SAVE META FILE HERE <<<
#with open(os.path.join(RUN_DIR, "meta.json"), "w") as f:
 #   json.dump(run_meta, f, indent=2)


def main():
    M_TOTAL = 30
    NUM_SLOTS = 12
    NUM_EPISODES = 30
    FRAMES_PER_EPISODE = 1000

    gamma_th_db = 0
    # if you used a manual seed somewhere, define it here once
    seed = 42
    seed_value = globals().get("seed", None)

    run_meta = {
        "M_total": int(M_TOTAL),
        "num_slots": int(NUM_SLOTS),
        "num_episodes": int(NUM_EPISODES),
        "frames_per_episode": int(FRAMES_PER_EPISODE) if "num_frames" in globals() else None,
        "seed": int(seed) if seed_value is not None else None,
        "torch_seed": int(torch.initial_seed()) if torch else None,
        "np_seed_state": str(np.random.get_state()[1][0]),
        "notes": "PPO AoI run"
    }

    RUN_DIR_SLT = make_run_dir(M_TOTAL, NUM_SLOTS, "SLT")

    RUN_DIR_RNDM = make_run_dir(M_TOTAL, NUM_SLOTS, "RNDM")

    out_dir = "Comparisons_RD_PPO"

    # Load data for PPO and Random
    data_ppo = load_episode_telemetry(RUN_DIR_SLT, filename=f"slotwise_dataU{M_TOTAL}S{NUM_SLOTS}.npy")
    data_random = load_episode_telemetry(RUN_DIR_RNDM, filename=f"slotwise_dataU{M_TOTAL}S{NUM_SLOTS}.npy")

    plot_time_averaged_system_aoi_comparison(data_ppo, data_random, NUM_SLOTS, FRAMES_PER_EPISODE, out_dir,
                                             out_pdf="system_aoi_time_avg_comparison.pdf")

    # Plot System Avg AoI Comparison for PPO vs Random
    #plot_system_avg_aoi_timewise_comparison(data_ppo, data_random, NUM_SLOTS, FRAMES_PER_EPISODE, NUM_EPISODES, out_dir)
    plot_system_avg_aoi_timewise_strict(
        data_ppo,
        data_random,
        NUM_SLOTS,
        FRAMES_PER_EPISODE,
        NUM_EPISODES,
        out_dir,
        out_pdf="Comparison_system_aoi_time_avg.pdf",
        also_plot_mean_of_user_mavgs=True,
        rolling_window=1,  # e.g., 1000 for a smoother auxiliary curve
        include_per_slot_in_main=True,  # True -> keep thin per-slot mean in the main figure
        save_avg_only=True,  # True -> also save a second "averages-only" figure
        avg_only_pdf="system_aoi_time_avg_only.pdf",
        avg_ylim_clip=(1, 99),  # y-axis clips by percentiles of average curves for better scale
    )

    plot_policy_analytics_modern_comparison(
        sar_log_dir_ppo=RUN_DIR_SLT,  # Directory for PPO data
        sar_log_dir_random=RUN_DIR_RNDM,  # Directory for Random data
        M_total=M_TOTAL,
        num_slots=NUM_SLOTS,
        out_dir=out_dir,
        n_bins_profile=30,
        roll_smooth=3,
        jitter=0.08,
        hex_gridsize=35,
        dpi=600
    )


if __name__ == "__main__":
    main()