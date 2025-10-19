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


def load_episode_telemetry(run_dir, filename="episode_data.npy"):
    """
    Load telemetry data from a .npy file stored in a run directory.
    """
    import os
    path = os.path.join(run_dir, filename)
    print(f"[LOAD] Loading episode data from {path}")
    return np.load(path, allow_pickle=True).item()


def plot_avg_aoi_per_user(data, num_slots, frames_per_episode, out_dir):
    import os, numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    uids = sorted(data.keys())

    for uid in uids:
        rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))

        t = []
        avg_aoi = []
        cum_aoi = 0.0

        for idx, r in enumerate(rows):
            # Global time index = ep * (frames * slots) + frame * slots + slot
            time_idx = r["ep"] * frames_per_episode * num_slots + r["frame"] * num_slots + r["slot"]
            aoi_val = r.get("aoi", 0.0)
            cum_aoi += aoi_val
            avg = cum_aoi / (idx + 1)

            t.append(time_idx)
            avg_aoi.append(avg)

        plt.figure()
        plt.plot(t, avg_aoi, marker='o', linestyle='-', label=f"U{uid}")
        plt.title(f"User {uid}: Moving Avg AoI Over Time")
        plt.xlabel("Slot Index")
        plt.ylabel("Avg AoI (Cumulative)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"user_{uid}_moving_avg_aoi.pdf"))
        plt.close()

def plot_system_avg_aoi(data, num_slots, frames_per_episode, out_dir, out_pdf="system_avg_aoi.pdf"):
    import os, numpy as np
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    uids = sorted(data.keys())
    num_episodes = max(r["ep"] for uid in uids for r in data[uid]) + 1
    total_slots_per_ep = num_slots * frames_per_episode

    system_avg_aoi_per_ep = []

    for ep in range(num_episodes):
        user_avg_list = []

        for uid in uids:
            rows = sorted([r for r in data[uid] if r["ep"] == ep], key=lambda r: (r["frame"], r["slot"]))
            if not rows:
                continue

            cum_aoi = 0.0
            avg_aoi_vals = []
            for idx, r in enumerate(rows):
                aoi_val = r.get("aoi", 0.0)
                cum_aoi += aoi_val
                avg_aoi_vals.append(cum_aoi / (idx + 1))

            # Take the final average AoI value at end of episode
            user_avg_list.append(avg_aoi_vals[-1])

        if user_avg_list:
            system_avg = sum(user_avg_list) / len(user_avg_list)
            system_avg_aoi_per_ep.append(system_avg)

    # Plotting
    plt.figure()
    plt.plot(range(num_episodes-1), system_avg_aoi_per_ep, marker='s', linestyle='--')
    plt.title("System Average AoI per Episode")
    plt.xlabel("Episode")
    plt.ylabel("System Avg AoI")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, out_pdf))
    plt.close()

    print(f"[PLOT] Saved system average AoI plot → {os.path.join(out_dir, out_pdf)}")
    return system_avg_aoi_per_ep

def plot_moving_avg_aoi_per_user(
    data, num_slots, frames_per_episode, num_episodes,
    out_dir, out_pdf="All_Users_MovingAvgAoI.pdf", episode_tick=5
):
    import os, math, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    uids = sorted(data.keys())
    ncols, nrows = 3, max(1, math.ceil(len(uids) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.1 * nrows), squeeze=False)
    axes = axes.ravel()

    T = num_slots * frames_per_episode
    episode_boundaries = [i * T for i in range(0, num_episodes + 1, episode_tick)]

    for ii, uid in enumerate(uids):
        rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        t, avg_aoi = [], []
        cum_aoi = 0.0

        for idx, r in enumerate(rows):
            time_idx = (r["ep"] - 1) * T + r["frame"] * num_slots + r["slot"]
            aoi_val = r.get("aoi", 0.0)
            cum_aoi += aoi_val
            avg = cum_aoi / (idx + 1)
            t.append(time_idx)
            avg_aoi.append(avg)

        ax = axes[ii]
        ax.plot(t, avg_aoi, color="purple", linestyle="-", linewidth=1.3, label=f"U{uid}")

        # Mark green hollow circle at every `episode_tick`
        mark_indices = [i for i, ti in enumerate(t) if ti in episode_boundaries]
        if mark_indices:
            ax.plot(
                [t[i] for i in mark_indices],
                [avg_aoi[i] for i in mark_indices],
                "o", markerfacecolor="none", markeredgecolor="green", markersize=5,
                linestyle="None"
            )

        ax.set_xticks(episode_boundaries)
        ax.set_xlim([0, T * num_episodes])
        ax.set_title(f"U{uid}", fontsize=8)
        ax.set_xlabel("Slot Index")
        ax.set_ylabel("Average AoI")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False, fontsize=7)

    for j in range(len(uids), nrows * ncols):
        axes[j].axis("off")

    fig.tight_layout(pad=0.6)
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved All Users Moving Avg AoI → {out_path}")

def plot_avg_aoi_per_user_separate(data, num_slots, frames_per_episode, num_episodes, out_dir, out_pdf_prefix="user_avg_aoi"):
    import os, math, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    total_time = num_slots * frames_per_episode * num_episodes
    tick_spacing = num_slots * frames_per_episode
    xticks = np.arange(0, total_time + 1, tick_spacing)

    uids = sorted(data.keys())
    ncols, nrows = 3, max(1, math.ceil(len(uids) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 2.1 * nrows), squeeze=False)
    axes = axes.ravel()

    for ii, uid in enumerate(uids):
        rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        if not rows:
            continue

        t = []
        avg_aoi = []
        cum_aoi = 0.0

        for idx, r in enumerate(rows):
            time_idx = (r["ep"] - 1) * frames_per_episode * num_slots + r["frame"] * num_slots + r["slot"]
            aoi_val = r.get("aoi", 0.0)
            cum_aoi += aoi_val
            avg = cum_aoi / (idx + 1)
            t.append(time_idx)
            avg_aoi.append(avg)

        ax = axes[ii]
        ax.plot(t, avg_aoi, marker='o', linestyle='-', label=f"U{uid}", linewidth=1.3)
        ax.set_title(f"U{uid}", fontsize=8)
        ax.set_xlabel("Slot Index")
        ax.set_ylabel("Moving Avg AoI")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(xticks)
        ax.set_xlim([0, total_time])
        ax.legend(loc="best", frameon=False, fontsize=7)

    for j in range(len(uids), nrows * ncols):
        axes[j].axis("off")

    fig.tight_layout(pad=0.6)
    out_path = os.path.join(out_dir, f"{out_pdf_prefix}.pdf")
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved per-user average AoI → {out_path}")


def plot_all_users_energy(telemetry, num_slots, frames_per_episode, out_pdf="Energy_All_Users.pdf", out_dir="telemetry_plots"):
    import os, math, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    def _flag(x):
        if x is None: return False
        try:
            xf = float(x)
            if np.isnan(xf): return False
            return xf > 0.5
        except Exception:
            return str(x).strip().lower() in ("1","true","t","yes","y")

    uids = sorted(telemetry.by_uid.keys())
    ncols, nrows = 3, max(1, math.ceil(len(uids) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3*ncols, 2.1*nrows), squeeze=False)
    axes = axes.ravel()

    for ii, uid in enumerate(uids):
        rows = sorted(telemetry.by_uid[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        if not rows:
            continue

        t    = np.array([(r["ep"]-1) * frames_per_episode * num_slots + r["frame"] * num_slots + r["slot"] for r in rows], float)
        bat  = np.array([np.nan if r.get("battery")   is None else float(r["battery"])   for r in rows])
        harv = np.array([np.nan if r.get("harvested") is None else float(r["harvested"]) for r in rows])
        reqf = np.array([_flag(r.get("required")) for r in rows], dtype=bool)
        sch  = np.array([0 if r.get("scheduled") is None else int(r["scheduled"]) for r in rows])
        role = np.array([r.get("pd_role", "IDLE") for r in rows], dtype=object)

        ax = axes[ii]
        ax.plot(t, bat,  label="battery",   color="tab:blue")
        ax.plot(t, harv, label="harvested", color="tab:green", alpha=0.85)

        # Red filled circles for energy used
        if np.any(reqf):
            ax.plot(t[reqf], bat[reqf], "ro", ms=3, label="energy used")

        # Hollow green markers for scheduling
        mH = (sch == 1) & (role == "NOMA-H")
        mL = (sch == 1) & (role == "NOMA-L")
        mO = (sch == 1) & (role == "OMA")
        if np.any(mH): ax.plot(t[mH], bat[mH], "^", ms=5.5, markerfacecolor='none', markeredgecolor="green", linestyle="None", label="PD-NOMA H")
        if np.any(mL): ax.plot(t[mL], bat[mL], "v", ms=5.5, markerfacecolor='none', markeredgecolor="green", linestyle="None", label="PD-NOMA L")
        if np.any(mO): ax.plot(t[mO], bat[mO], "s", ms=5.0, markerfacecolor='none', markeredgecolor="green", linestyle="None", label="Single OMA")

        ax.grid(True, alpha=0.3)
        ax.set_title(f"U{uid}", fontsize=8)
        ax.set_xlabel("Slot Index"); ax.set_ylabel("Energy Timeline (J)")

        # Tick every 10 slots
        ax.set_xticks(np.arange(0, int(t.max()) + 1, 10))
        ax.set_xlim([0, int(t.max())])

        #ax.set_xticks(np.arange(0, t.max() + 1, 10))
        ax.legend(loc="best", frameon=False, fontsize=7)

    for j in range(len(uids), nrows*ncols):
        axes[j].axis("off")

    fig.tight_layout(pad=0.6)
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved energy timelines → {out_path}")

def plot_all_users_aoi(telemetry, num_slots, frames_per_episode, out_pdf="AOI_All_Users.pdf", out_dir="telemetry_plots"):
    import os, math, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    uids = sorted(telemetry.by_uid.keys())
    ncols, nrows = 3, max(1, math.ceil(len(uids) / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3*ncols, 2.1*nrows), squeeze=False)
    axes = axes.ravel()

    for ii, uid in enumerate(uids):
        rows = sorted(telemetry.by_uid[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        if not rows:
            continue

        # Use true global slot index: ep * frames * slots + frame * slots + slot
        t = np.array([(r["ep"]-1) * num_slots * frames_per_episode + r["frame"] * num_slots + r["slot"] for r in rows], int)
        #t = np.array([r["ep"] * frames_per_episode * num_slots + r["frame"] * num_slots + r["slot"] for r in rows], float)
        aoi = np.array([r.get("aoi", np.nan) for r in rows], float)
        dec = np.array([np.nan if r.get("decoded") is None else float(r["decoded"]) for r in rows])
        sch = np.array([0 if r.get("scheduled") is None else int(r["scheduled"]) for r in rows])
        role = np.array([r.get("pd_role", "IDLE") for r in rows], dtype=object)

        ax = axes[ii]
        ax.plot(t, aoi, color="tab:blue", linewidth=1.5, label="AoI")

        # decoded success marker
        ok = (dec == 1.0)
        if np.any(ok):
            ax.plot(t[ok], aoi[ok], "r*", ms=6, label="decoded=1")

        # green hollow markers where scheduled
        mH = (sch == 1) & (role == "NOMA-H")
        mL = (sch == 1) & (role == "NOMA-L")
        mO = (sch == 1) & (role == "OMA")
        if np.any(mH): ax.plot(t[mH], aoi[mH], "^", ms=6, markerfacecolor='none', markeredgecolor="green", linestyle="None", label="PD-NOMA H")
        if np.any(mL): ax.plot(t[mL], aoi[mL], "v", ms=6, markerfacecolor='none', markeredgecolor="green", linestyle="None", label="PD-NOMA L")
        if np.any(mO): ax.plot(t[mO], aoi[mO], "s", ms=5.5, markerfacecolor='none', markeredgecolor="green", linestyle="None", label="Single OMA")

        ax.grid(True, alpha=0.3)
        ax.set_title(f"U{uid}", fontsize=8)
        ax.set_xlabel("Slot Index"); ax.set_ylabel("AoI Timeline")

        # Ticks every 10 slots
        ax.set_xticks(np.arange(0, int(t.max()) + 1, 10))
        ax.set_xlim([0, int(t.max())])

        #ax.set_xticks(np.arange(0, t.max() + 1, 10))

        ax.legend(loc="best", frameon=False, fontsize=7)

    for j in range(len(uids), nrows*ncols):
        axes[j].axis("off")

    fig.tight_layout(pad=0.6)
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved AoI timelines → {out_path}")

def plot_time_averaged_system_aoi(data, num_slots, frames_per_episode, out_dir, out_pdf="system_aoi_time_avg.pdf"):
    import os, numpy as np, matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "lines.linewidth": 1.1, "grid.linewidth": 0.5,
    })
    os.makedirs(out_dir, exist_ok=True)

    uids = sorted(data.keys())
    num_episodes = max(r["ep"] for uid in uids for r in data[uid]) + 1
    T = num_slots * frames_per_episode  # total slots per episode

    system_avg = []

    for ep in range(num_episodes):
        total = 0.0
        count = 0
        for uid in uids:
            aoi_vals = [r["aoi"] for r in data[uid] if r["ep"] == ep]
            total += sum(aoi_vals)
            count += len(aoi_vals)
        avg = total / count if count > 0 else np.nan
        system_avg.append(avg)

    plt.figure(figsize=(4.5, 2.6))
    x = np.arange(1, num_episodes + 1)
    plt.plot(x, system_avg, color="purple", marker="o", markerfacecolor="none",
             markeredgecolor="green", linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("System Avg AoI (Time-Averaged)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, out_pdf), dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved time-averaged system AoI → {out_pdf}")

def plot_slotwise_rewards(sar_log, out_dir="telemetry_plots", window=100):
    import os, pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl


    sar_log_path = os.path.join(sar_log, fr"sar_logU{M_total}S{num_slots}.pkl")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    with open(sar_log_path, "rb") as f:
        sar_data = pickle.load(f)

    rewards = [entry["reward"] for entry in sar_data if "reward" in entry]
    steps = list(range(len(rewards)))

    def moving_avg(data, k):
        return np.convolve(data, np.ones(k) / k, mode='valid')

    ma_rewards = moving_avg(rewards, window)
    ma_steps = steps[window - 1:]

    # Combined Plot
    plt.figure(figsize=(6, 2.5))
    plt.plot(steps, rewards, color="purple", alpha=0.3, label="Raw Reward (faded)")
    plt.plot(ma_steps, ma_rewards, color="purple", linewidth=1.3, label=f"Moving Avg (k={window})")
    plt.xlabel("Slot Index")
    plt.ylabel("Reward")
    plt.title("Reward Convergence")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_convergence.pdf"), dpi=600)
    plt.close()
    print(f"[PLOT] Reward convergence plot saved → {out_dir}")

def plot_system_avg_aoi_timewise(data, num_slots, frames_per_episode, out_dir, out_pdf="system_aoi_time_avg.pdf"):
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    uids = sorted(data.keys())
    system_avg = []
    t = []
    total_aoi = 0.0
    time_index = 0

    all_rows = []
    for uid in uids:
        all_rows.extend(data[uid])
    all_rows.sort(key=lambda r: (r["ep"], r["frame"], r["slot"]))

    for r in all_rows:
        aoi_sum = sum(r.get("aoi", 0.0) for r in all_rows if r["ep"] <= r["ep"] and r["frame"] <= r["frame"])
        total_aoi += r.get("aoi", 0.0)
        time_index += 1
        avg = total_aoi / (len(uids) * time_index)
        t.append(time_index)
        system_avg.append(avg)

    plt.figure(figsize=(6, 2.5))
    plt.plot(t, system_avg, color="tab:blue", linewidth=1.3)
    plt.xlabel("Slot Index")
    plt.ylabel("System Avg AoI (Cumulative)")
    plt.title("System AoI Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, out_pdf), dpi=600)
    plt.close()
    print(f"[PLOT] System AoI over time saved → {out_pdf}")

# ---------- State-Action AoI Analysis ----------
def plot_state_action_pair_aoi(
    sar_log_dir,
    M_total,
    num_slots,
    out_dir="telemetry_plots",
    n_bins_1d=25,
    n_bins_2d=30,
    clip_percentiles=(0.5, 99.5),
    figure_dpi=150,
):
    """
    Visualize which actions are taken at which AoI levels (AoI_near=state[0], AoI_far=state[1])
    from SAR pickle: sar_logU{M_total}S{num_slots}.pkl

    Parameters
    ----------
    sar_log_dir : str
        Directory containing SAR logs (same param as in reward plot).
    M_total : int
        Total users (used in filename pattern).
    num_slots : int
        Number of slots (used in filename pattern).
    out_dir : str
        Directory to save figures.
    n_bins_1d, n_bins_2d, clip_percentiles, figure_dpi : tuning params
    """
    import os, pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import BoundaryNorm
    import matplotlib.ticker as mticker

    # ---------- 0) Build correct SAR path ----------
    sar_log_path = os.path.join(sar_log_dir, f"sar_logU{M_total}S{num_slots}.pkl")
    if not os.path.exists(sar_log_path):
        raise FileNotFoundError(f"SAR file not found at: {sar_log_path}")

    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) Aesthetics ----------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    # ---------- 2) Load SAR ----------
    with open(sar_log_path, "rb") as f:
        sar = pickle.load(f)

    states, actions, rewards = [], [], []

    def _push(s, a, r):
        states.append(np.asarray(s))
        actions.append(int(a))
        rewards.append(float(r if r is not None else 0.0))

    if isinstance(sar, dict):
        s_key = next((k for k in sar.keys() if k.lower() in ["s", "state", "states"]), None)
        a_key = next((k for k in sar.keys() if k.lower() in ["a", "action", "actions"]), None)
        r_key = next((k for k in sar.keys() if k.lower() in ["r", "reward", "rewards"]), None)
        S, A, R = sar[s_key], sar[a_key], sar[r_key]
        for i in range(min(len(S), len(A), len(R))):
            _push(S[i], A[i], R[i])
    elif isinstance(sar, (list, tuple)):
        for item in sar:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                _push(item[0], item[1], item[2] if len(item) > 2 else 0.0)
            elif isinstance(item, dict):
                s = item.get("s", item.get("state"))
                a = item.get("a", item.get("action"))
                r = item.get("r", item.get("reward", 0.0))
                if s is not None and a is not None:
                    _push(s, a, r)
    else:
        raise TypeError("Unsupported SAR structure — must be dict or list of (s,a,r).")

    S = np.vstack(states)
    A = np.array(actions, dtype=int)

    if S.ndim != 2 or S.shape[1] < 2:
        raise ValueError(f"State shape invalid: {S.shape} (need at least 2 dims for AoI_near, AoI_far)")

    # ---------- 3) Extract AoI_near & AoI_far ----------
    AoI_n = S[:, 0].astype(float)
    AoI_f = S[:, 1].astype(float)

    mask = np.isfinite(AoI_n) & np.isfinite(AoI_f)
    AoI_n, AoI_f, A = AoI_n[mask], AoI_f[mask], A[mask]

    unique_actions = np.unique(A)
    action_to_idx = {act: i for i, act in enumerate(unique_actions)}

    # ---------- Helper for 1D heatmap ----------
    def _build_heat(x, n_bins):
        lo, hi = np.nanpercentile(x, clip_percentiles)
        edges = np.linspace(lo, hi, n_bins + 1)
        heat = np.zeros((len(unique_actions), n_bins), float)
        idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        for b, act in zip(idx, A):
            heat[action_to_idx[act], b] += 1
        colsum = heat.sum(axis=0, keepdims=True)
        heat = np.divide(heat, np.maximum(colsum, 1e-12), where=(colsum > 0))
        return heat, edges

    # ---------- 4) Plot AoI_near Heatmap ----------
    heat_n, edges_n = _build_heat(AoI_n, n_bins_1d)
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=figure_dpi)
    im = ax.imshow(
        heat_n, aspect="auto", origin="lower",
        extent=[edges_n[0], edges_n[-1], -0.5, len(unique_actions)-0.5],
        interpolation="nearest",
    )
    ax.set_yticks(range(len(unique_actions)))
    ax.set_yticklabels([f"a={a}" for a in unique_actions])
    ax.set_xlabel("AoI_near (state[0])"); ax.set_ylabel("Action")
    ax.set_title("AoI_near → Action (P(action|AoI bin))")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Probability")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"state_action_AoIn_U{M_total}S{num_slots}.pdf"))
    plt.close(fig)

    # ---------- 5) Plot AoI_far Heatmap ----------
    heat_f, edges_f = _build_heat(AoI_f, n_bins_1d)
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=figure_dpi)
    im = ax.imshow(
        heat_f, aspect="auto", origin="lower",
        extent=[edges_f[0], edges_f[-1], -0.5, len(unique_actions)-0.5],
        interpolation="nearest",
    )
    ax.set_yticks(range(len(unique_actions)))
    ax.set_yticklabels([f"a={a}" for a in unique_actions])
    ax.set_xlabel("AoI_far (state[1])"); ax.set_ylabel("Action")
    ax.set_title("AoI_far → Action (P(action|AoI bin))")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Probability")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"state_action_AoIf_U{M_total}S{num_slots}.pdf"))
    plt.close(fig)

    # ---------- 6) 2D Decision Map ----------
    lo_n, hi_n = np.nanpercentile(AoI_n, clip_percentiles)
    lo_f, hi_f = np.nanpercentile(AoI_f, clip_percentiles)
    edges_n2 = np.linspace(lo_n, hi_n, n_bins_2d + 1)
    edges_f2 = np.linspace(lo_f, hi_f, n_bins_2d + 1)

    counts = np.zeros((len(unique_actions), n_bins_2d, n_bins_2d))
    idx_n = np.clip(np.digitize(AoI_n, edges_n2) - 1, 0, n_bins_2d - 1)
    idx_f = np.clip(np.digitize(AoI_f, edges_f2) - 1, 0, n_bins_2d - 1)

    for i_n, i_f, act in zip(idx_n, idx_f, A):
        counts[action_to_idx[act], i_n, i_f] += 1

    totals = counts.sum(axis=0)
    majority = np.argmax(counts, axis=0)
    majority_mask = np.where(totals > 0, majority, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=figure_dpi)
    im = ax.imshow(
        majority_mask.T, origin="lower",
        extent=[edges_n2[0], edges_n2[-1], edges_f2[0], edges_f2[-1]],
        aspect="auto", interpolation="nearest",
    )
    norm = BoundaryNorm(np.arange(-0.5, len(unique_actions)+0.5), len(unique_actions))
    im.set_norm(norm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(len(unique_actions)))
    cbar.ax.yaxis.set_major_formatter(mticker.FixedFormatter([f"a={a}" for a in unique_actions]))
    ax.set_xlabel("AoI_near (state[0])"); ax.set_ylabel("AoI_far (state[1])")
    ax.set_title("Most Likely Action (AoI_near vs AoI_far)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"state_action_decision_U{M_total}S{num_slots}.pdf"))
    plt.close(fig)

    print(f"[State–Action AoI] Done: {sar_log_path}")

def plot_policy_analytics_modern(
    sar_log_dir,
    M_total,
    num_slots,
    out_dir="telemetry_plots",
    n_bins_profile=30,
    roll_smooth=3,        # moving average smoothing over adjacent bins (>=1)
    jitter=0.08,          # raincloud jitter (x-direction)
    hex_gridsize=35,      # hexbin resolution per panel
    dpi=150,
):
    """
    Modern, interpretable policy visuals (no heatmaps):
      1) Policy profiles with confidence bands: P(a | AoI_near), P(a | AoI_far)
      2) Raincloud distributions (violin + box + jitter) for AoI_near and AoI_far per action
      3) Hex-multiples density: (AoI_near, AoI_far) points for each action

    Loads: {sar_log_dir}/sar_logU{M_total}S{num_slots}.pkl
    Saves : *_U{M_total}S{num_slots}.pdf under out_dir
    """
    import os, pickle, math
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # ---------- Paths & style ----------
    os.makedirs(out_dir, exist_ok=True)
    sar_log_path = os.path.join(sar_log_dir, f"sar_logU{M_total}S{num_slots}.pkl")
    if not os.path.exists(sar_log_path):
        raise FileNotFoundError(f"Missing SAR log: {sar_log_path}")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix", "axes.unicode_minus": False, "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    # ---------- Load SAR ----------
    with open(sar_log_path, "rb") as f:
        sar = pickle.load(f)

    states, actions = [], []
    def _push(s, a): states.append(np.asarray(s)); actions.append(int(a))
    if isinstance(sar, dict):
        s_key = next(k for k in sar if k.lower() in ["s", "state", "states"])
        a_key = next(k for k in sar if k.lower() in ["a", "action", "actions"])
        S, A = sar[s_key], sar[a_key]
        for i in range(min(len(S), len(A))): _push(S[i], A[i])
    elif isinstance(sar, (list, tuple)):
        for item in sar:
            if isinstance(item, (list, tuple)) and len(item) >= 2: _push(item[0], item[1])
            elif isinstance(item, dict):
                s = item.get("s", item.get("state")); a = item.get("a", item.get("action"))
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
    actions_unique = np.sort(np.unique(A))
    K = len(actions_unique)

    # ---------- helper: binned profile with SE bands ----------
    def _profile(x, a, n_bins):
        # Bin by quantiles to balance data per bin (more stable probabilities)
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(x, qs)
        # ensure strictly increasing edges
        edges = np.unique(edges)
        if len(edges) - 1 < max(5, n_bins//2):
            # fallback uniform if too many ties
            edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        p = np.zeros(len(centers))
        n = np.zeros(len(centers), int)
        for i in range(len(centers)):
            m = (x >= edges[i]) & (x < edges[i+1]) if i < len(centers)-1 else (x >= edges[i]) & (x <= edges[i+1])
            n[i] = m.sum()
            if n[i] > 0:
                p[i] = (A[m] == a).mean()
        # smooth by simple moving average if desired
        if roll_smooth > 1:
            w = roll_smooth
            def ma(v):
                if len(v) < w: return v
                pad = w//2
                vv = np.pad(v, (pad, pad), mode="edge")
                out = np.convolve(vv, np.ones(w)/w, mode="valid")
                return out[:len(v)]
            p = ma(p); n = np.maximum(ma(n.astype(float)), 1e-12)

        # binomial 95% CI via normal approx
        se = np.sqrt(np.maximum(p * (1 - p) / np.maximum(n, 1), 1e-12))
        lo, hi = np.clip(p - 1.96 * se, 0, 1), np.clip(p + 1.96 * se, 0, 1)
        return centers, p, lo, hi, n

    # ---------- (1) Policy profiles with confidence bands ----------
    def _policy_profiles(x, x_label, tag):
        fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=dpi)
        for a in actions_unique:
            c, p, lo, hi, n = _profile(x, a, n_bins_profile)
            ax.plot(c, p, label=f"a={a}", linewidth=1.4)
            ax.fill_between(c, lo, hi, alpha=0.18, linewidth=0)
        ax.set_xlabel(x_label); ax.set_ylabel("P(action | AoI)")
        ax.set_title(f"Policy Profiles vs {x_label}")
        ax.grid(True, alpha=0.35)
        ax.legend(title="Actions", ncols=min(K, 4), fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"policy_profiles_{tag}_U{M_total}S{num_slots}.pdf"),
                    format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    _policy_profiles(AoI_n, "AoI_near (state[0])", "AoIn")
    _policy_profiles(AoI_f, "AoI_far (state[1])", "AoIf")

    # ---------- (2) Rainclouds (violin + box + jitter) ----------
    def _raincloud(x, x_label, tag):
        # Build per-action lists
        groups = [x[A == a] for a in actions_unique]
        fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=dpi)
        parts = ax.violinplot(groups, showmeans=False, showmedians=False, showextrema=False)
        # soften violin faces
        for pc in parts['bodies']:
            pc.set_alpha(0.28)
        # box markers
        for i, g in enumerate(groups, start=1):
            if len(g) == 0: continue
            q1, med, q3 = np.percentile(g, [25, 50, 75])
            ax.plot([i-0.15, i+0.15], [med, med], lw=1.6)               # median line
            ax.plot([i, i], [q1, q3], lw=3.0, alpha=0.9)                # IQR bar
            # jitter points (subsample large sets)
            idx = np.arange(len(g))
            if len(idx) > 800: idx = np.random.choice(idx, 800, replace=False)
            xjit = i + (np.random.rand(len(idx)) - 0.5) * jitter
            ax.plot(xjit, g[idx], "o", ms=1.6, alpha=0.35, linestyle="None")
        ax.set_xticks(range(1, K+1)); ax.set_xticklabels([f"a={a}" for a in actions_unique])
        ax.set_xlabel("Action"); ax.set_ylabel(x_label)
        ax.set_title(f"Raincloud: {x_label} by Action")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"raincloud_{tag}_U{M_total}S{num_slots}.pdf"),
                    format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    _raincloud(AoI_n, "AoI_near (state[0])", "AoIn")
    _raincloud(AoI_f, "AoI_far (state[1])", "AoIf")

    # ---------- (3) Hex-multiples density per action ----------
    # Small multiples layout (<=3 columns)
    ncols = min(3, K); nrows = math.ceil(K / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 2.8*nrows), dpi=dpi, squeeze=False)
    for i, a in enumerate(actions_unique):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        m = (A == a)
        if m.sum() > 0:
            hb = ax.hexbin(AoI_n[m], AoI_f[m], gridsize=hex_gridsize, mincnt=1)
            cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("Count")
        ax.set_title(f"Action a={a}", fontsize=8)
        ax.set_xlabel("AoI_near (state[0])"); ax.set_ylabel("AoI_far (state[1])")
        ax.grid(True, alpha=0.2)
    # empty panels off
    for j in range(K, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle("Hex Density of (AoI_near, AoI_far) per Action", y=0.995, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, f"hex_multiples_U{M_total}S{num_slots}.pdf"),
                format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("[PLOT] Saved:",
          os.path.join(out_dir, f"policy_profiles_AoIn_U{M_total}S{num_slots}.pdf"),
          os.path.join(out_dir, f"policy_profiles_AoIf_U{M_total}S{num_slots}.pdf"),
          os.path.join(out_dir, f"raincloud_AoIn_U{M_total}S{num_slots}.pdf"),
          os.path.join(out_dir, f"raincloud_AoIf_U{M_total}S{num_slots}.pdf"),
          os.path.join(out_dir, f"hex_multiples_U{M_total}S{num_slots}.pdf"))

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
    data,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir,
    out_pdf="system_aoi_time_avg.pdf",
    also_plot_mean_of_user_mavgs=True,
    rolling_window=None   # e.g., 1000 for a smoother auxiliary curve
):
    """
    Builds:
      - per-slot mean AoI across users at each global t (fast, O(N))
      - running mean over time (AAoI-like system convergence)
      - optional: mean of users' moving averages at each t (aligned & forward-filled)

    Saves a compact plot showing the curves and episode boundaries.
    """
    import os, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix", "axes.unicode_minus": False, "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.1, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    uids = sorted(data.keys())
    U = len(uids)
    T = num_slots * frames_per_episode
    T_total = T * num_episodes

    # --- A) Per-slot mean AoI across users (cross-sectional)
    sum_per_t = {}
    cnt_per_t = {}
    for uid in uids:
        rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        for r in rows:
            t = (r["ep"] - 1) * T + r["frame"] * num_slots + r["slot"]
            aoi = float(r.get("aoi", 0.0))
            sum_per_t[t] = sum_per_t.get(t, 0.0) + aoi
            cnt_per_t[t] = cnt_per_t.get(t, 0) + 1

    if not sum_per_t:
        print("[PLOT] No data found; nothing to plot.")
        return

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

    # Running (cumulative) mean over time (AAoI-like)
    valid = np.isfinite(per_slot_mean)
    ps = np.where(valid, per_slot_mean, 0.0)
    w = np.where(valid, 1.0, 0.0)
    csum = np.cumsum(ps)
    wsum = np.cumsum(w)
    system_running_mean = np.divide(csum, np.maximum(wsum, 1e-12))

    roll_curve = None
    if rolling_window and rolling_window > 1:
        k = int(rolling_window)
        kernel = np.ones(k)/k
        num = np.convolve(ps, kernel, mode="same")
        den = np.convolve(w, kernel, mode="same")
        roll_curve = np.divide(num, np.maximum(den, 1e-12))

    # --- B) Mean of users' moving averages (built from exact per-user MAvgs)
    mean_of_user_mavgs = None
    if also_plot_mean_of_user_mavgs:
        user_series = compute_user_moving_avgs(data, num_slots, frames_per_episode, num_episodes)
        # Build aligned array by forward-filling each user's series onto [0..max_t]
        mtx = np.full((U, max_t + 1), np.nan, dtype=float)
        for ui, uid in enumerate(uids):
            t_u = user_series[uid]["t"]
            m_u = user_series[uid]["mavg"]
            if t_u.size == 0:
                continue
            # place values and forward-fill
            mtx_row = np.full(max_t + 1, np.nan, dtype=float)
            mtx_row[t_u] = m_u
            last = np.nan
            for i in range(max_t + 1):
                if np.isfinite(mtx_row[i]):
                    last = mtx_row[i]
                else:
                    mtx_row[i] = last
            mtx[ui] = mtx_row
        mean_of_user_mavgs = np.nanmean(mtx, axis=0)  # mean across users at each t

    # --- Plot
    x = np.arange(max_t + 1)
    episode_boundaries = [i * T for i in range(0, num_episodes + 1)]

    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    # thin: instantaneous per-slot mean across users
    ax.plot(x, per_slot_mean, linewidth=0.9, alpha=0.35, label="Per-slot mean AoI (across users)")
    # thick: system running mean (AAoI-like)
    ax.plot(x, system_running_mean, linewidth=1.6, label="Running mean AoI (system)")
    if roll_curve is not None:
        ax.plot(x, roll_curve, linestyle="--", linewidth=1.2, label=f"Rolling mean (w={rolling_window})")
    if mean_of_user_mavgs is not None:
        ax.plot(x, mean_of_user_mavgs, linewidth=1.2, label="Mean of users' moving avgs")

    # episode markers
    for eb in episode_boundaries:
        if eb <= max_t:
            ax.axvline(eb, color="0.88", linewidth=0.7, zorder=0)

    ax.set_xlim(0, max_t)
    ax.set_xlabel("Slot Index")
    ax.set_ylabel("AoI")
    ax.set_title("System Average AoI Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, ncols=2)
    fig.tight_layout()

    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] System AoI over time saved → {out_path}")

def make_run_dir(M_total, num_slots):
    #stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"AoI_U{M_total}_S{num_slots}_P"
    out = os.path.join(name)
    os.makedirs(out, exist_ok=True)
    return out


# ------------------- Environment params (yours) -------------------
num_slots          = 5
frames_per_episode = 1000
num_episodes       = 30
M_total            = 15

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

RUN_DIR = make_run_dir(M_total, num_slots)


# 4. >>> SAVE META FILE HERE <<<
#with open(os.path.join(RUN_DIR, "meta.json"), "w") as f:
 #   json.dump(run_meta, f, indent=2)

print(f"[SAVE] Run dir created: {RUN_DIR}")


data = load_episode_telemetry(RUN_DIR, filename=f"slotwise_dataU{M_total}S{num_slots}.npy")
#plot_all_users_avg_aoi_combined(data, num_slots=num_slots, frames_per_episode=frames_per_episode, num_episodes = num_episodes ,out_dir="AoI_U6_S2_PPNR")
plot_moving_avg_aoi_per_user(
    data=data,
    num_slots=num_slots,
    frames_per_episode=frames_per_episode,
    num_episodes=num_episodes,
    out_dir=RUN_DIR,
    out_pdf="All_Users_MovingAvgAoICluster2.pdf",
    episode_tick=5
)


plot_system_avg_aoi(data, num_slots=num_slots, frames_per_episode=frames_per_episode,  out_dir=RUN_DIR, out_pdf="system_avg_aoicluster2.pdf")

plot_slotwise_rewards(RUN_DIR, out_dir=RUN_DIR, window=1000)

#plot_time_averaged_system_aoi(data, num_slots, frames_per_episode, RUN_DIR, out_pdf="system_aoi_time_avg.pdf")

plot_state_action_pair_aoi(
    sar_log_dir=RUN_DIR,
    M_total=M_total,
    num_slots=num_slots,
    out_dir=RUN_DIR,
    n_bins_1d=25,
    n_bins_2d=30,
    clip_percentiles=(0.5, 99.5),
    figure_dpi=600
)

plot_policy_analytics_modern(
    sar_log_dir=RUN_DIR,
    M_total=M_total,
    num_slots=num_slots,
    out_dir=RUN_DIR,

    n_bins_profile=20,   # try 25–50
    roll_smooth=30 ,
    dpi= 600# set 1 to disable smoothing

)


plot_system_avg_aoi_timewise_strict(
    data=data,
    num_slots=num_slots,
    frames_per_episode=frames_per_episode,
    num_episodes=num_episodes,
    out_dir=RUN_DIR,
    out_pdf="system_aoi_time_avg.pdf",
    also_plot_mean_of_user_mavgs=True,
    rolling_window=1,   # optional
)


#plot_system_avg_aoi_timewise(data, num_slots, frames_per_episode, RUN_DIR, out_pdf="system_aoi_time.pdf")

#plot_all_users_aoi(telemetry, num_slots, frames_per_episode,  out_pdf="AOI_All_Users.pdf", out_dir=RUN_DIR)
#plot_all_users_energy(telemetry, num_slots, frames_per_episode, out_pdf="Energy_All_Users.pdf", out_dir=RUN_DIR)
