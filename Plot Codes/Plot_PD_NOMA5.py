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

'''''''''
def plot_episode_reward_curves_from_sar(
    sar_log_dir,
    M_total,
    num_slots,
    frames_per_episode,
    out_dir=None,
    normalize_mode="global",   # "global" (min-max over all episodes) or "running"
    ma_window=10,              # moving-average window over episodes for visual smoothing
):
    """
    From slot-wise SAR log, compute per-episode reward metrics and plot:
      (a) Raw sum of reward per episode
      (b) Raw average reward per episode (per slot)
      (c) Normalized average reward per episode (like many papers' Fig. 3)

    Assumes one episode has `frames_per_episode * num_slots` slots.
    Saves 3 PDFs in out_dir (defaults to sar_log_dir).
    """
    import os, pickle, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

    # ---------- paths ----------
    if out_dir is None:
        out_dir = sar_log_dir
    os.makedirs(out_dir, exist_ok=True)
    sar_log_path = os.path.join(sar_log_dir, f"sar_logU{M_total}S{num_slots}.pkl")
    if not os.path.exists(sar_log_path):
        raise FileNotFoundError(f"Missing SAR log: {sar_log_path}")

    # ---------- style ----------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    # ---------- load SAR (slot-wise rewards) ----------
    with open(sar_log_path, "rb") as f:
        sar = pickle.load(f)

    # robust extraction of slot rewards
    rewards = []
    if isinstance(sar, dict):
        # common patterns: list of dicts under a key, or flat arrays
        # Try flat vector first
        for k in ("reward", "rewards", "r"):
            if k in sar and np.ndim(sar[k]) >= 1:
                rewards = np.asarray(sar[k], dtype=float).tolist()
                break
        if not rewards:
            # try list of entries
            seq_key = next((k for k in sar if isinstance(sar[k], (list, tuple))), None)
            if seq_key is not None:
                for entry in sar[seq_key]:
                    if isinstance(entry, dict) and ("reward" in entry or "r" in entry):
                        rewards.append(float(entry.get("reward", entry.get("r"))))
    elif isinstance(sar, (list, tuple)):
        for item in sar:
            if isinstance(item, dict) and ("reward" in item or "r" in item):
                rewards.append(float(item.get("reward", item.get("r"))))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                # Possible (s, a, r, ...)
                try:
                    rewards.append(float(item[2]))
                except Exception:
                    pass
    else:
        raise TypeError(f"Unsupported SAR format: {type(sar)}")

    if not rewards:
        raise ValueError("Could not find slot-wise rewards in SAR log.")

    rewards = np.asarray(rewards, dtype=float)
    slots_per_episode = int(num_slots) * int(frames_per_episode)
    total_slots = len(rewards)
    n_episodes = total_slots // slots_per_episode
    remainder = total_slots % slots_per_episode
    if n_episodes == 0:
        raise ValueError(f"Not enough slots ({total_slots}) for one episode of {slots_per_episode} slots.")
    if remainder != 0:
        # Truncate the tail to keep full episodes
        rewards = rewards[: n_episodes * slots_per_episode]

    # reshape into [episodes, slots_per_episode]
    R = rewards.reshape(n_episodes, slots_per_episode)

    # ---------- per-episode metrics ----------
    ep_sum = R.sum(axis=1)                 # raw sum per episode
    ep_avg = R.mean(axis=1)                # raw average per-slot reward per episode

    # normalized average reward per episode
    if normalize_mode == "running":
        # running min-max normalization (online)
        ep_norm = np.zeros_like(ep_avg)
        run_min, run_max = np.inf, -np.inf
        for i, v in enumerate(ep_avg):
            run_min = min(run_min, v)
            run_max = max(run_max, v)
            denom = max(run_max - run_min, 1e-8)
            ep_norm[i] = (v - run_min) / denom
    else:
        # "global" min-max over all episodes (common for plots)
        lo, hi = np.min(ep_avg), np.max(ep_avg)
        denom = max(hi - lo, 1e-8)
        ep_norm = (ep_avg - lo) / denom

    # ---------- moving-average helper (over episodes) ----------
    def moving_avg(y, k):
        k = int(max(1, k))
        if k == 1 or len(y) < k:
            return y, np.arange(len(y))
        ma = np.convolve(y, np.ones(k)/k, mode="valid")
        x = np.arange(k-1, k-1+len(ma))
        return ma, x

    ep_idx = np.arange(1, n_episodes + 1)

    # ---------- (a) Raw sum per episode ----------
    plt.figure(figsize=(6.6, 3.2))
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

    # ---------- (b) Raw average per episode ----------
    plt.figure(figsize=(6.6, 3.2))
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

    # ---------- (c) Normalized average per episode (Fig. 3-style) ----------
    plt.figure(figsize=(6.6, 3.2))
    plt.plot(ep_idx, ep_norm, alpha=0.35, label=f"Normalized Avg ({normalize_mode})")
    if ma_window and n_episodes >= ma_window:
        ma, x = moving_avg(ep_norm, ma_window)
        plt.plot(x+1, ma, linewidth=1.6, label=f"Moving Avg (k={ma_window})")
    plt.xlabel("Episode"); plt.ylabel("Normalized Avg Reward [0–1]")
    plt.title("Per-Episode Normalized Average Reward")
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    out_c = os.path.join(out_dir, f"episode_reward_avg_normalized_{normalize_mode}_U{M_total}S{num_slots}.pdf")
    plt.savefig(out_c, dpi=600, bbox_inches="tight"); plt.close()

    print(f"[PLOT] Saved:\n  {out_a}\n  {out_b}\n  {out_c}")

    # return arrays if you want to compute early-stop externally
    return {
        "episode_index": ep_idx,
        "episode_sum": ep_sum,
        "episode_avg": ep_avg,
        "episode_avg_normalized": ep_norm,
    }
'''
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
):
    """
    IEEE-style figures for state–action vs AoI analysis.
    Saves three PNGs:
      - state_action_AoIn_U{M}S{S}.png
      - state_action_AoIf_U{M}S{S}.png
      - state_action_decision_U{M}S{S}.png
    """
    import os, pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import BoundaryNorm
    import matplotlib.ticker as mticker

    # ---------- 0) Paths ----------
    sar_log_path = os.path.join(sar_log_dir, f"sar_logU{M_total}S{num_slots}.pkl")
    if not os.path.exists(sar_log_path):
        raise FileNotFoundError(f"SAR file not found at: {sar_log_path}")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1) IEEE aesthetics ----------
    IEEE_WIDTH  = 3.4   # inches
    IEEE_HEIGHT = 2.1   # inches
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
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

    # ---------- 2) Load SAR ----------
    with open(sar_log_path, "rb") as f:
        sar = pickle.load(f)

    states, actions, rewards = [], [], []

    def _push(s, a, r):
        states.append(np.asarray(s))
        actions.append(int(a))
        rewards.append(float(0.0 if r is None else r))

    if isinstance(sar, dict):
        s_key = next((k for k in sar if k.lower() in ["s", "state", "states"]), None)
        a_key = next((k for k in sar if k.lower() in ["a", "action", "actions"]), None)
        r_key = next((k for k in sar if k.lower() in ["r", "reward", "rewards"]), None)
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
        raise ValueError(f"State shape invalid: {S.shape} (need ≥2 dims for AoI_near, AoI_far)")

    # ---------- 3) Extract AoIs ----------
    AoI_n = S[:, 0].astype(float)
    AoI_f = S[:, 1].astype(float)
    mask = np.isfinite(AoI_n) & np.isfinite(AoI_f)
    AoI_n, AoI_f, A = AoI_n[mask], AoI_f[mask], A[mask]

    unique_actions = np.unique(A)
    action_to_idx = {act: i for i, act in enumerate(unique_actions)}

    # ---------- Helper: 1D heat over AoI bins ----------
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

    # ---------- 4) AoI_near → Action ----------
    heat_n, edges_n = _build_heat(AoI_n, n_bins_1d)
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    im = ax.imshow(
        heat_n, aspect="auto", origin="lower",
        extent=[edges_n[0], edges_n[-1], -0.5, len(unique_actions)-0.5],
        interpolation="nearest",
    )
    ax.set_yticks(range(len(unique_actions)))
    ax.set_yticklabels([f"$a={a}$" for a in unique_actions])
    ax.set_xlabel(r"AoI$_{\text{near}}$")
    ax.set_ylabel("Action")
    # no title (IEEE compact)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Probability")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"state_action_AoIn_U{M_total}S{num_slots}.png"),
                dpi=600, format="png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 5) AoI_far → Action ----------
    heat_f, edges_f = _build_heat(AoI_f, n_bins_1d)
    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    im = ax.imshow(
        heat_f, aspect="auto", origin="lower",
        extent=[edges_f[0], edges_f[-1], -0.5, len(unique_actions)-0.5],
        interpolation="nearest",
    )
    ax.set_yticks(range(len(unique_actions)))
    ax.set_yticklabels([f"$s_m={a}$" for a in unique_actions])
    ax.set_xlabel(r"AoI")
    ax.set_ylabel("Slot Assignment")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("Probability")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"state_action_AoIf_U{M_total}S{num_slots}.png"),
                dpi=600, format="png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 6) 2D decision map: most-likely action over (AoI_near, AoI_far) ----------
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

    fig, ax = plt.subplots(figsize=(IEEE_WIDTH, IEEE_HEIGHT * 1.2))
    im = ax.imshow(
        majority_mask.T, origin="lower",
        extent=[edges_n2[0], edges_n2[-1], edges_f2[0], edges_f2[-1]],
        aspect="auto", interpolation="nearest",
    )
    norm = BoundaryNorm(np.arange(-0.5, len(unique_actions) + 0.5), len(unique_actions))
    im.set_norm(norm)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04,
                        ticks=np.arange(len(unique_actions)))
    cbar.ax.yaxis.set_major_formatter(
        mticker.FixedFormatter([f"$a={a}$" for a in unique_actions])
    )
    ax.set_xlabel(r"AoI$_{\text{near}}$")
    ax.set_ylabel(r"AoI$_{\text{far}}$")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"state_action_decision_U{M_total}S{num_slots}.png"),
                dpi=600, format="png", bbox_inches="tight")
    plt.close(fig)

    print(f"[State–Action AoI] Done: {sar_log_path}")


def analyze_system_success_history_from_sar(
    sar_log_dir,
    M_total,
    num_slots,
    frames_per_episode,
    core_state_len=12,          # number of non-history features at the start of each state
    hist_len=None,              # if None, infer as len(state) - core_state_len
    out_dir=None,
    ma_window=10                # moving-average over episodes for smoother curves
):
    """
    Analyze the 'last-K system success rates' history embedded in SAR state vectors.

    Produces:
      (A) Per-episode avg of the most-recent success rate (hist[-1])  + MA curve
      (B) Per-episode avg of the full history window mean             + MA curve
      (C) Heatmap: episode (y) × history lag 0..K-1 (x) of avg success (mean over slots)
      (D) If rewards are logged: scatter of reward vs recent success with Pearson r

    Assumptions:
      - SAR log is a pickle with a list of dict entries, each with keys:
        {"ep", "frame", "slot", "state", "action", "reward"(optional)}
      - 'state' is a 1D vector: [core_state_len features] + [K history elements]
      - One episode = frames_per_episode * num_slots slots
    """
    import os, pickle, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

    # ---------- paths ----------
    if out_dir is None:
        out_dir = sar_log_dir
    os.makedirs(out_dir, exist_ok=True)
    sar_log_path = os.path.join(sar_log_dir, f"sar_logU{M_total}S{num_slots}.pkl")
    if not os.path.exists(sar_log_path):
        raise FileNotFoundError(f"Missing SAR log: {sar_log_path}")

    # ---------- style ----------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    # ---------- load SAR ----------
    with open(sar_log_path, "rb") as f:
        sar = pickle.load(f)

    if not isinstance(sar, (list, tuple)):
        # support a dict with 'logs' etc.
        if isinstance(sar, dict):
            # look for a list-like field
            seq_key = next((k for k in sar if isinstance(sar[k], (list, tuple))), None)
            if seq_key is not None:
                sar = sar[seq_key]
        if not isinstance(sar, (list, tuple)):
            raise TypeError(f"Unsupported SAR structure: {type(sar)}")

    # ---------- extract states, rewards ----------
    states = []
    rewards = []
    for entry in sar:
        if not isinstance(entry, dict):
            continue
        st = entry.get("state", None)
        if st is None:
            continue
        states.append(np.asarray(st, dtype=float))
        if "reward" in entry and entry["reward"] is not None:
            try:
                rewards.append(float(entry["reward"]))
            except Exception:
                rewards.append(np.nan)
        else:
            rewards.append(np.nan)

    if len(states) == 0:
        raise ValueError("No 'state' vectors found in SAR log.")

    states = np.asarray(states, dtype=float)  # shape [Nslots_total, state_dim]
    rewards = np.asarray(rewards, dtype=float)

    state_dim = states.shape[1]
    if hist_len is None:
        hist_len = int(state_dim - core_state_len)
        if hist_len <= 0:
            raise ValueError(
                f"hist_len inferred as {hist_len}. "
                f"Check core_state_len={core_state_len} and state_dim={state_dim}."
            )

    # ---------- slice out history window ----------
    hist = states[:, core_state_len: core_state_len + hist_len]  # [N, K]
    recent = hist[:, -1]                                         # most-recent success rate per slot
    hist_mean = np.nanmean(hist, axis=1)                         # mean over the K window per slot

    # (optional) slope of the K-history per slot (linear fit): +ve => rising success
    # x = [0..K-1], slope per slot
    xh = np.arange(hist_len, dtype=float)
    xh_centered = xh - xh.mean()
    denom = np.sum(xh_centered**2) if hist_len > 1 else 1.0
    slopes = np.sum((hist - hist.mean(axis=1, keepdims=True)) * xh_centered, axis=1) / max(denom, 1e-12)

    # ---------- reshape into episodes ----------
    slots_per_episode = int(num_slots) * int(frames_per_episode)
    total_slots = hist.shape[0]
    n_episodes = total_slots // slots_per_episode
    if n_episodes == 0:
        raise ValueError(f"Not enough slots ({total_slots}) for one episode of {slots_per_episode} slots.")
    # truncate tail to full episodes
    upto = n_episodes * slots_per_episode
    recent = recent[:upto]
    hist_mean = hist_mean[:upto]
    slopes = slopes[:upto]
    rewards = rewards[:upto]
    hist = hist[:upto, :]

    recent_ep   = recent.reshape(n_episodes, slots_per_episode).mean(axis=1)
    hmean_ep    = hist_mean.reshape(n_episodes, slots_per_episode).mean(axis=1)
    slope_ep    = slopes.reshape(n_episodes, slots_per_episode).mean(axis=1)
    reward_ep   = np.nanmean(rewards.reshape(n_episodes, slots_per_episode), axis=1)

    # average history profile per episode (mean across slots)
    hist_ep = hist.reshape(n_episodes, slots_per_episode, hist_len).mean(axis=1)  # [episodes, K]

    # ---------- moving average helper ----------
    def moving_avg(y, k):
        k = int(max(1, k))
        if k == 1 or len(y) < k:
            return y, np.arange(len(y))
        ma = np.convolve(y, np.ones(k)/k, mode="valid")
        x = np.arange(k-1, k-1+len(ma))
        return ma, x

    ep_idx = np.arange(1, n_episodes + 1)

    # ---------- (A) recent success rate per episode ----------
    plt.figure(figsize=(6.6, 3.0))
    plt.plot(ep_idx, recent_ep, alpha=0.35, label="Recent success (hist[-1])")
    if ma_window and n_episodes >= ma_window:
        ma, x = moving_avg(recent_ep, ma_window)
        plt.plot(x+1, ma, linewidth=1.6, label=f"Moving Avg (k={ma_window})")
    plt.xlabel("Episode"); plt.ylabel("Recent System Success")
    plt.title("Per-Episode Recent Success Rate (from history tail)")
    plt.grid(True, alpha=0.3); plt.legend(frameon=False, fontsize=8)
    out_a = os.path.join(out_dir, f"sys_succ_recent_U{M_total}S{num_slots}.pdf")
    plt.tight_layout(); plt.savefig(out_a, dpi=600, bbox_inches="tight"); plt.close()

    # ---------- (B) mean of history window per episode ----------
    plt.figure(figsize=(6.6, 3.0))
    plt.plot(ep_idx, hmean_ep, alpha=0.35, label="Mean over history window")
    if ma_window and n_episodes >= ma_window:
        ma, x = moving_avg(hmean_ep, ma_window)
        plt.plot(x+1, ma, linewidth=1.6, label=f"Moving Avg (k={ma_window})")
    plt.xlabel("Episode"); plt.ylabel("Mean Success over K")
    plt.title("Per-Episode Mean of Success History Window")
    plt.grid(True, alpha=0.3); plt.legend(frameon=False, fontsize=8)
    out_b = os.path.join(out_dir, f"sys_succ_histmean_U{M_total}S{num_slots}.pdf")
    plt.tight_layout(); plt.savefig(out_b, dpi=600, bbox_inches="tight"); plt.close()

    # ---------- (C) Heatmap of avg history profile by episode ----------
    plt.figure(figsize=(7.2, 3.4))
    im = plt.imshow(hist_ep, aspect="auto", origin="lower",
                    extent=[0, hist_len-1, 1, n_episodes], interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Avg success (per episode)")
    plt.xlabel("History lag (0=oldest, K-1=most recent)")
    plt.ylabel("Episode")
    plt.title("Average System Success History Profile by Episode")
    out_c = os.path.join(out_dir, f"sys_succ_hist_heatmap_U{M_total}S{num_slots}.pdf")
    plt.tight_layout(); plt.savefig(out_c, dpi=600, bbox_inches="tight"); plt.close()

    # ---------- (D) Optional: reward vs recent success ----------
    if not np.isnan(reward_ep).all():
        # Pearson correlation
        valid = np.isfinite(reward_ep) & np.isfinite(recent_ep)
        corr = np.corrcoef(reward_ep[valid], recent_ep[valid])[0,1] if valid.any() else np.nan
        plt.figure(figsize=(6.0, 3.0))
        plt.scatter(recent_ep[valid], reward_ep[valid], s=10, alpha=0.4)
        plt.xlabel("Recent Success (episode avg of hist[-1])")
        plt.ylabel("Episode Avg Reward")
        plt.title(f"Reward vs Recent Success (r={corr:.2f})")
        plt.grid(True, alpha=0.3)
        out_d = os.path.join(out_dir, f"reward_vs_recent_succ_U{M_total}S{num_slots}.pdf")
        plt.tight_layout(); plt.savefig(out_d, dpi=600, bbox_inches="tight"); plt.close()
    else:
        out_d = None

    # ---------- simple text diagnostics ----------
    pos = np.mean(slope_ep > 0.0)
    neg = np.mean(slope_ep < 0.0)
    print(f"[HIST] K={hist_len}  core={core_state_len}  episodes={n_episodes}")
    print(f"[HIST] Fraction of episodes with increasing success profile slope: {pos:.2f}")
    print(f"[HIST] Fraction with decreasing slope: {neg:.2f}")
    print(f"[SAVE] History plots →\n  {out_a}\n  {out_b}\n  {out_c}" + (f"\n  {out_d}" if out_d else ""))

    return {
        "episode_index": ep_idx,
        "recent_success_ep": recent_ep,
        "hist_mean_ep": hmean_ep,
        "hist_profile_ep": hist_ep,  # shape [episodes, K]
        "slope_ep": slope_ep,
        "reward_ep": reward_ep,
    }


def plot_policy_analytics_modern(
    sar_log_dir,
    M_total,
    num_slots,
    out_dir=None,              # if None, saves alongside SAR log dir
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
    Colorful policy analytics:
      1) Policy profiles with bands: P(a | AoI_near), P(a | AoI_far)
      2) Rainclouds (violin + box + jitter) by action
      3) Hex-multiples (focused, log-scaled, de-noised)
      4) Decision strips (sorted AoI, colored by action)
      5) Smoothed 2D policy contours (frontiers)
    Saves in the SAME directory as the SAR log by default.
    """
    import os, pickle, math
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm

    # ---------- Paths ----------
    if out_dir is None:
        out_dir = sar_log_dir
    os.makedirs(out_dir, exist_ok=True)
    sar_log_path = os.path.join(sar_log_dir, f"sar_logU{M_total}S{num_slots}.pkl")
    if not os.path.exists(sar_log_path):
        raise FileNotFoundError(f"Missing SAR log: {sar_log_path}")

    # ---------- Global style ----------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.3, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    # ---------- Color palette ----------
    # A long, vibrant set of distinct, publication-safe colors (cycled if K > len base)
    BASE_COLORS = [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
        "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
        "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    ]
    def distinct_colors(K):
        if K <= len(BASE_COLORS):
            return BASE_COLORS[:K]
        cols = []
        i = 0
        while len(cols) < K:
            cols.append(BASE_COLORS[i % len(BASE_COLORS)])
            i += 1
        return cols

    BAND_ALPHA = 0.18
    ACCENT_DARK = "#4B4453"  # titles/lines; neutral-dark

    def _lbl_near():
        return r"AoI$_{\mathrm{near}}$" if pretty_labels else "Near-user AoI"
    def _lbl_far():
        return r"AoI$_{\mathrm{far}}$"  if pretty_labels else "Far-user AoI"

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
    COLORS = distinct_colors(K)
    act2idx = {a:i for i,a in enumerate(actions_unique)}
    def col_of(a): return COLORS[act2idx[a]]

    # ---------- helpers ----------
    def _profile(x, a, n_bins):
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(x, qs)
        edges = np.unique(edges)
        if len(edges) - 1 < max(5, n_bins//2):
            edges = np.linspace(np.min(x), np.max(x), n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        p = np.zeros(len(centers))
        n = np.zeros(len(centers), int)
        for i in range(len(centers)):
            m = (x >= edges[i]) & (x < edges[i+1]) if i < len(centers)-1 else (x >= edges[i]) & (x <= edges[i+1])
            n[i] = m.sum()
            if n[i] > 0:
                p[i] = (A[m] == a).mean()
        if roll_smooth > 1 and len(centers) >= roll_smooth:
            w = roll_smooth; pad = w//2
            def _ma(v):
                vv = np.pad(v, (pad, pad), mode="edge")
                out = np.convolve(vv, np.ones(w)/w, mode="valid")
                return out[:len(v)]
            p = _ma(p); n = np.maximum(_ma(n.astype(float)), 1e-12)
        se = np.sqrt(np.maximum(p * (1 - p) / np.maximum(n, 1), 1e-12))
        lo, hi = np.clip(p - 1.96 * se, 0, 1), np.clip(p + 1.96 * se, 0, 1)
        return centers, p, lo, hi

    def _robust_limits(x):
        lo, hi = np.nanpercentile(x, xclip_quantiles)
        if not np.isfinite(lo): lo = np.nanmin(x)
        if not np.isfinite(hi): hi = np.nanmax(x)
        if lo == hi: hi = lo + 1e-9
        return lo, hi

    # ---------- (1) Policy profiles ----------
    '''''''''
    def _policy_profiles(x, x_label, tag):
        fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=dpi)
        for a in actions_unique:
            c, p, lo, hi = _profile(x, a, n_bins_profile)
            ax.plot(c, p, color=col_of(a), label=f"a={a}", linewidth=1.8)
            ax.fill_between(c, lo, hi, color=col_of(a), alpha=BAND_ALPHA, linewidth=0)
        ax.set_xlabel(x_label); ax.set_ylabel(r"$P(\mathrm{action}\mid \mathrm{AoI})$")
        ax.set_title(f"Policy Profiles vs {x_label}", color=ACCENT_DARK)
        ax.grid(True, alpha=0.35)
        ax.legend(title="Actions", ncols=min(K, 4), fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"policy_profiles_{tag}_U{M_total}S{num_slots}.pdf"),
                    format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    _policy_profiles(AoI_n, _lbl_near(), "AoIn")
    _policy_profiles(AoI_f, _lbl_far(),  "AoIf")
     '''''

    # --- pooled AoI profile (NEAR + FAR together) ---
    AoI_all = np.concatenate([AoI_n, AoI_f])
    A_all = np.concatenate([A, A])  # duplicate actions to pair with both AoIs

    # --- generic binning helper that lets us pass the action array explicitly ---
    def _profile_with_actions(x, action_arr, a, n_bins):
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
                p[i] = (action_arr[m] == a).mean()
        if roll_smooth > 1 and len(centers) >= roll_smooth:
            w = roll_smooth;
            pad = w // 2

            def _ma(v):
                vv = np.pad(v, (pad, pad), mode="edge")
                out = np.convolve(vv, np.ones(w) / w, mode="valid")
                return out[:len(v)]

            p = _ma(p);
            n = np.maximum(_ma(n.astype(float)), 1e-12)
        se = np.sqrt(np.maximum(p * (1 - p) / np.maximum(n, 1), 1e-12))
        lo, hi = np.clip(p - 1.96 * se, 0, 1), np.clip(p + 1.96 * se, 0, 1)
        return centers, p, lo, hi

    # --- a single plotting function; pass whichever action array you want ---
    def _policy_profile(x, action_arr, x_label, filename_suffix, title_prefix="Policy Profile"):
        fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=dpi)
        for a in actions_unique:
            c, p, lo, hi = _profile_with_actions(x, action_arr, a, n_bins_profile)
            ax.plot(c, p, color=col_of(a), label=f"a={a}", linewidth=1.8)
            ax.fill_between(c, lo, hi, color=col_of(a), alpha=BAND_ALPHA, linewidth=0)
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$P(\mathrm{action}\mid \mathrm{AoI})$")
        ax.set_title(f"{title_prefix} vs {x_label}", color=ACCENT_DARK)
        ax.grid(True, alpha=0.35)
        ax.legend(title="Actions", ncols=min(K, 4), fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{filename_suffix}_U{M_total}S{num_slots}.pdf"),
                    format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    AoI_all = np.concatenate([AoI_n, AoI_f])
    A_all = np.concatenate([A, A])
    _policy_profile(AoI_all, A_all, "AoI", "policy_profile_pooled",
                    title_prefix="Policy Profile")

    # ---------- (2) Rainclouds ----------
    def _raincloud(x, x_label, tag):
        groups = [x[A == a] for a in actions_unique]
        fig, ax = plt.subplots(figsize=(7.6, 3.8), dpi=dpi)
        parts = ax.violinplot(groups, showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(COLORS[i])
            pc.set_edgecolor('none')
            pc.set_alpha(0.28)
        for i, g in enumerate(groups, start=1):
            if len(g) == 0: continue
            q1, med, q3 = np.percentile(g, [25, 50, 75])
            ax.plot([i-0.15, i+0.15], [med, med], lw=1.9, color=COLORS[i-1])  # median
            ax.plot([i, i], [q1, q3], lw=3.0, alpha=0.95, color=COLORS[i-1])  # IQR
            idx = np.arange(len(g))
            if len(idx) > 800: idx = np.random.choice(idx, 800, replace=False)
            xjit = i + (np.random.rand(len(idx)) - 0.5) * jitter
            ax.plot(xjit, g[idx], "o", ms=1.6, alpha=0.35, linestyle="None", color=COLORS[i-1])
        ax.set_xticks(range(1, K+1)); ax.set_xticklabels([f"a={a}" for a in actions_unique])
        ax.set_xlabel("Action"); ax.set_ylabel(x_label)
        ax.set_title(f"Raincloud: {x_label} by Action", color=ACCENT_DARK)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"raincloud_{tag}_U{M_total}S{num_slots}.pdf"),
                    format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    _raincloud(AoI_n, _lbl_near(), "AoIn")
    _raincloud(AoI_f, _lbl_far(),  "AoIf")

    # ---------- (3) Hex-multiples (focused, denoised) ----------
    ncols = min(3, K); nrows = math.ceil(K / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 2.8*nrows), dpi=dpi, squeeze=False)
    for i, a in enumerate(actions_unique):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        m = (A == a)
        if m.sum() > 0:
            xv, yv = AoI_n[m], AoI_f[m]
            xlo, xhi = _robust_limits(xv)
            ylo, yhi = _robust_limits(yv)
            # colorful, perceptually-uniform colormap for density
            hb = ax.hexbin(
                xv, yv,
                gridsize=hex_gridsize,
                mincnt=max(1, int(hex_mincnt)),
                norm=LogNorm(),
                cmap="viridis",
            )
            counts = hb.get_array()
            if counts.size:
                thr = np.percentile(counts, hex_min_percentile)
                hb.set_clim(vmin=max(thr, 1))   # hide low-density bins
                cmap = hb.get_cmap().copy()
                cmap.set_under(alpha=0.0)
                hb.set_cmap(cmap)
            ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_title(f"Action a={a}", fontsize=8, color=ACCENT_DARK)
        ax.set_xlabel(_lbl_near()); ax.set_ylabel(_lbl_far())
        ax.grid(True, alpha=0.2)
    for j in range(K, nrows*ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
    fig.suptitle("Hex Density of (AoI_near, AoI_far) per Action", y=0.995, fontsize=9, color=ACCENT_DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, f"hex_multiples_U{M_total}S{num_slots}.pdf"),
                format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ---------- (5) Smoothed 2D policy contours ----------
    if make_policy_contours and K >= 2:
        def _gaussian_kernel1d(sigma, radius):
            x = np.arange(-radius, radius+1, dtype=float)
            k = np.exp(-(x*x)/(2*sigma*sigma))
            k /= k.sum()
            return k
        def _blur2d(arr, sigma):
            if sigma <= 0: return arr
            radius = int(max(1, round(3*sigma)))
            k = _gaussian_kernel1d(sigma, radius)
            tmp = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, arr)
            out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, tmp)
            return out

        def _edges(v):
            lo, hi = _robust_limits(v)
            return np.linspace(lo, hi, contour_bins+1)

        xe, ye = _edges(AoI_n), _edges(AoI_f)
        ix = np.clip(np.digitize(AoI_n, xe)-1, 0, contour_bins-1)
        iy = np.clip(np.digitize(AoI_f, ye)-1, 0, contour_bins-1)

        counts = np.zeros((K, contour_bins, contour_bins), dtype=float)
        for cx, cy, a in zip(ix, iy, A):
            counts[act2idx[a], cx, cy] += 1.0

        probs = np.zeros_like(counts)
        for k_idx in range(K):
            probs[k_idx] = _blur2d(counts[k_idx], contour_sigma)
        denom = np.maximum(probs.sum(axis=0), 1e-12)
        probs /= denom

        Xc = 0.5*(xe[:-1] + xe[1:])
        Yc = 0.5*(ye[:-1] + ye[1:])
        Xg, Yg = np.meshgrid(Xc, Yc, indexing="ij")

        # region coloring with action colors
        region = np.argmax(probs, axis=0)  # [X,Y] in {0..K-1}
        region_cmap = ListedColormap(COLORS)
        norm = BoundaryNorm(np.arange(-0.5, K+0.5, 1), K)

        fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=dpi)
        ax.contourf(Xg, Yg, region.T, levels=np.arange(-0.5, K+0.5, 1),
                    cmap=region_cmap, norm=norm, alpha=0.35)

        # draw p=0.5 contours per action in that action's line color
        for k_idx, a in enumerate(actions_unique):
            try:
                ax.contour(Xg, Yg, probs[k_idx].T, levels=[0.5],
                           colors=[col_of(a)], linewidths=1.6)
            except Exception:
                pass

        ax.set_xlabel(_lbl_near()); ax.set_ylabel(_lbl_far())
        ax.set_title("Smoothed Policy Frontiers", color=ACCENT_DARK)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"policy_contours_U{M_total}S{num_slots}.pdf"),
                    format="pdf", dpi=600, bbox_inches="tight")
        plt.close(fig)

    print("[PLOT] Saved to:", out_dir)

'''''''''
def plot_episode_reward_curves_from_sar(
    sar_log_dir,
    M_total,
    num_slots,
    frames_per_episode,
    out_dir=None,
    normalize_mode="global",   # "global" | "running"
    ma_window=10,
    fig3_mode="cumulative",    # "cumulative" (paper-like) | "per-episode"
):
    """
    From slot-wise SAR log, compute per-episode reward metrics and plot:
      (a) Raw sum per episode
      (b) Raw average per episode (per slot)
      (c) Normalized 'Fig. 3'-style curve:
          - cumulative: normalized average *cumulative* reward (default)
          - per-episode: normalized per-episode average reward

    Assumes one episode has frames_per_episode * num_slots slots.
    Saves 3 PDFs in out_dir (defaults to sar_log_dir).
    """
    import os, pickle, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

    # ---------- paths ----------
    if out_dir is None:
        out_dir = sar_log_dir
    os.makedirs(out_dir, exist_ok=True)
    sar_log_path = os.path.join(sar_log_dir, f"sar_logU{M_total}S{num_slots}.pkl")
    if not os.path.exists(sar_log_path):
        raise FileNotFoundError(f"Missing SAR log: {sar_log_path}")

    # ---------- style ----------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    # ---------- load & canonicalize to chronological records ----------
    with open(sar_log_path, "rb") as f:
        sar = pickle.load(f)

    # Expect SARLogger.logs = list of dicts with ep, frame, slot, reward
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
        # Fallback: flat vector with implicit order (not recommended, but supported)
        rewards = None
        for k in ("reward", "rewards", "r"):
            if isinstance(sar, dict) and k in sar:
                rewards = np.asarray(sar[k], dtype=float)
                break
        if rewards is None:
            raise TypeError(f"Unsupported SAR format: {type(sar)}")
        records = [(-1, -1, i, rewards[i]) for i in range(len(rewards))]

    # Sort by (episode, frame, slot) to guarantee chronological order
    records.sort(key=lambda t: (t[0], t[1], t[2]))

    # Validate rewards
    none_idx = [i for i, rec in enumerate(records) if rec[3] is None]
    if none_idx:
        raise ValueError(f"{len(none_idx)} SAR entries have reward=None. Fill/update rewards before plotting.")

    rewards = np.asarray([float(rec[3]) for rec in records], dtype=float)

    # ---------- episode shaping ----------
    slots_per_episode = int(num_slots) * int(frames_per_episode)
    if slots_per_episode <= 0:
        raise ValueError("slots_per_episode must be positive.")
    total_slots = len(rewards)
    n_episodes = total_slots // slots_per_episode
    remainder = total_slots % slots_per_episode
    if n_episodes == 0:
        raise ValueError(f"Not enough slots ({total_slots}) for one episode of {slots_per_episode} slots.")
    if remainder:
        print(f"[WARN] Truncating {remainder} trailing slots (partial episode dropped).")
        rewards = rewards[: n_episodes * slots_per_episode]

    R = rewards.reshape(n_episodes, slots_per_episode)

    # ---------- metrics ----------
    ep_sum = R.sum(axis=1)   # (a) raw sum per episode
    ep_avg = R.mean(axis=1)  # (b) raw average per-slot reward per episode

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

    # (c) Fig-3 style base series
    if fig3_mode == "cumulative":
        base = np.cumsum(ep_avg) / np.arange(1, n_episodes + 1)
        ylabel = "Normalized Avg CUMULATIVE Reward [0–1]"
        title_c = "Fig-3 Style Normalized Cumulative Reward"
    elif fig3_mode == "per-episode":
        base = ep_avg.copy()
        ylabel = "Normalized Avg Reward [0–1]"
        title_c = "Fig-3 Style Normalized Per-Episode Reward"
    else:
        raise ValueError("fig3_mode must be 'cumulative' or 'per-episode'.")

    ep_norm = running_norm(base) if normalize_mode == "running" else minmax_norm(base)

    # ---------- moving-average helper (over episodes) ----------
    def moving_avg(y, k):
        k = int(max(1, k))
        if k == 1 or len(y) < k:
            return y, np.arange(len(y))
        ma = np.convolve(y, np.ones(k)/k, mode="valid")
        x = np.arange(k-1, k-1+len(ma))
        return ma, x

    ep_idx = np.arange(1, n_episodes + 1)

    # ---------- (a) Raw sum per episode ----------
    plt.figure(figsize=(6.6, 3.2))
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

    # ---------- (b) Raw average per episode ----------
    plt.figure(figsize=(6.6, 3.2))
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
    plt.figure(figsize=(6.6, 3.2))
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

    print(f"[PLOT] Saved:\n  {out_a}\n  {out_b}\n  {out_c}")

    return {
        "episode_index": ep_idx,
        "episode_sum": ep_sum,
        "episode_avg": ep_avg,
        "normalized_curve": ep_norm,  # Fig-3 style output
    }
'''

def plot_reward_and_success_from_sar(
    sar_log_dir,
    M_total,
    num_slots,
    frames_per_episode,
    out_dir=None,
    normalize_mode="global",   # "global" | "running" over episodes
    ma_window=10,
    filename=None,             # if None -> f"sar_logU{M_total}S{num_slots}.pkl"
    strict_rewards=False       # True => raise if any reward is None; False => skip reward plot
):
    """
    Make two episode-level learning curves from a single SAR file:
      (1) Per-episode *normalized* average reward (non-cumulative)
      (2) Per-episode success rate (derived from state.system_succ_hist's last value)

    Notes on success extraction:
      - Your state = concat([12-dim core], system_succ_hist)
      - We take the LAST element of system_succ_hist as the 'latest' success proxy.
      - We average last-success across slots -> frame; then across frames -> episode.
    """
    import os, pickle, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

    # ---------- paths ----------
    if out_dir is None:
        out_dir = sar_log_dir
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        filename = f"sar_logU{M_total}S{num_slots}.pkl"
    sar_path = os.path.join(sar_log_dir, filename)
    if not os.path.exists(sar_path):
        raise FileNotFoundError(f"Missing SAR log: {sar_path}")

    # ---------- style ----------
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.8, "lines.linewidth": 1.2, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })

    # ---------- load & canonicalize ----------
    with open(sar_path, "rb") as f:
        sar = pickle.load(f)

    records = []
    # Expect list of dicts produced by SARLogger.log(ep, frame, slot, state, action[, reward])
    if isinstance(sar, list):
        for it in sar:
            if isinstance(it, dict) and all(k in it for k in ("ep", "frame", "slot")):
                records.append((
                    int(it["ep"]), int(it["frame"]), int(it["slot"]),
                    it.get("state"), it.get("action"), it.get("reward", it.get("r"))
                ))
            elif isinstance(it, (list, tuple)) and len(it) >= 5:
                # (ep, frame, slot, state, action, [reward])
                ep, fr, sl = int(it[0]), int(it[1]), int(it[2])
                state, action = it[3], it[4]
                reward = it[5] if len(it) > 5 else None
                records.append((ep, fr, sl, state, action, reward))
    elif isinstance(sar, dict) and "logs" in sar:
        for it in sar["logs"]:
            records.append((
                int(it["ep"]), int(it["frame"]), int(it["slot"]),
                it.get("state"), it.get("action"), it.get("reward", it.get("r"))
            ))
    else:
        raise TypeError(f"Unsupported SAR format: {type(sar)}")

    if not records:
        raise ValueError("SAR log is empty.")

    # chronological order
    records.sort(key=lambda t: (t[0], t[1], t[2]))

    # ---------- extract rewards & success-from-state ----------
    # Rewards (may contain None)
    rewards = np.array([ (float(r[5]) if r[5] is not None else np.nan) for r in records ], dtype=float)

    # Success: from state's system_succ_hist last value
    # state was saved as list/np array; core is 12-dim, rest is hist (len = K)
    latest_success = []
    for (_, _, _, st, _, _) in records:
        if st is None:
            latest_success.append(np.nan)
            continue
        s = np.asarray(st, dtype=float).ravel()
        if s.size < 13:
            latest_success.append(np.nan)
            continue
        # deduce K = len(s) - 12; take last element:
        last_succ = s[-1]
        # clamp to [0,1] just in case
        if not np.isfinite(last_succ):
            last_succ = np.nan
        else:
            last_succ = float(np.clip(last_succ, 0.0, 1.0))
        latest_success.append(last_succ)
    latest_success = np.array(latest_success, dtype=float)

    # ---------- episode shaping ----------
    slots_per_episode = int(num_slots) * int(frames_per_episode)
    total_slots = len(records)
    n_episodes = total_slots // slots_per_episode
    rem = total_slots % slots_per_episode
    if n_episodes == 0:
        raise ValueError(f"Not enough slots ({total_slots}) for one episode of {slots_per_episode} slots.")
    if rem:
        print(f"[WARN] Truncating {rem} trailing slots (partial episode dropped).")
        rewards = rewards[: n_episodes*slots_per_episode]
        latest_success = latest_success[: n_episodes*slots_per_episode]

    # reshape to [episodes, frames, slots]
    try:
        rewards_efs = rewards.reshape(n_episodes, frames_per_episode, num_slots)
        succ_efs    = latest_success.reshape(n_episodes, frames_per_episode, num_slots)
    except Exception as e:
        raise ValueError(f"Reshape failed; check num_slots/frames_per_episode. {e}")

    # Episode-average reward (per-slot)
    # If rewards have NaNs (missing), we’ll handle according to strict_rewards.
    ep_avg_reward = np.nanmean(rewards_efs, axis=(1,2))  # mean over frames & slots

    # Per-frame success = mean over slots of last-success; then per-episode = mean over frames
    frame_success = np.nanmean(succ_efs, axis=2)        # [episodes, frames]
    ep_success    = np.nanmean(frame_success, axis=1)   # [episodes]

    # ---------- normalization helpers ----------
    def minmax_norm(y):
        lo, hi = np.nanmin(y), np.nanmax(y)
        return (y - lo) / max(hi - lo, 1e-8)
    def running_norm(y):
        out = np.zeros_like(y, dtype=float)
        run_min, run_max = np.inf, -np.inf
        for i, v in enumerate(y):
            if not np.isfinite(v):
                out[i] = np.nan
                continue
            run_min = min(run_min, v); run_max = max(run_max, v)
            out[i] = (v - run_min) / max(run_max - run_min, 1e-8)
        return out

    # validate rewards if strict
    n_missing_rewards = int(np.isnan(ep_avg_reward).sum())
    if strict_rewards and n_missing_rewards > 0:
        raise ValueError(f"{n_missing_rewards} episodes have NaN avg reward (missing per-slot rewards).")

    # Normalize per-episode (non-cumulative) reward
    if np.all(np.isnan(ep_avg_reward)):
        ep_avg_reward_norm = None  # skip plot
    else:
        base = ep_avg_reward.copy()
        if normalize_mode == "running":
            ep_avg_reward_norm = running_norm(base)
        else:
            ep_avg_reward_norm = minmax_norm(base)

    # ---------- moving average ----------
    def moving_avg(y, k):
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(y)
        y_valid = y[mask]
        if k <= 1 or y_valid.size < k:
            # align indexes to original episodes where finite
            return y_valid, np.flatnonzero(mask)
        ma = np.convolve(y_valid, np.ones(k)/k, mode="valid")
        # place x at the end positions of the valid subsequence
        start = np.flatnonzero(mask)[0]
        x = np.arange(start + k - 1, start + k - 1 + len(ma))
        return ma, x

    ep_idx = np.arange(1, n_episodes + 1)

    # ---------- (1) Per-episode normalized average reward (non-cumulative) ----------
    out_reward = None
    if ep_avg_reward_norm is not None:
        plt.figure(figsize=(6.6, 3.2))
        plt.plot(ep_idx, ep_avg_reward_norm, alpha=0.4, label="Per-episode Avg Reward (normalized)")
        if ma_window and n_episodes >= ma_window:
            ma, x = moving_avg(ep_avg_reward_norm, ma_window)
            if len(ma) > 0:
                plt.plot(x+1, ma, linewidth=1.6, label=f"Moving Avg (k={ma_window})")
        plt.xlabel("Episode"); plt.ylabel("Normalized Avg Reward [0–1]")
        plt.title("Per-Episode Normalized Average Reward (Non-Cumulative)")
        plt.grid(True, alpha=0.3); plt.legend(fontsize=7, frameon=False)
        plt.tight_layout()
        out_reward = os.path.join(out_dir, f"per_episode_norm_avg_reward_U{M_total}S{num_slots}.pdf")
        plt.savefig(out_reward, dpi=600, bbox_inches="tight"); plt.close()
    else:
        print("[INFO] Reward plot skipped (no finite rewards).")

    # ---------- (2) Per-episode success rate ----------
    plt.figure(figsize=(6.6, 3.2))
    plt.plot(ep_idx, ep_success, alpha=0.4, label="Per-episode Success Rate (from state hist)")
    if ma_window and n_episodes >= ma_window:
        ma, x = moving_avg(ep_success, ma_window)
        if len(ma) > 0:
            plt.plot(x+1, ma, linewidth=1.6, label=f"Moving Avg (k={ma_window})")
    plt.xlabel("Episode"); plt.ylabel("Success Rate (0–1)")
    plt.title("Per-Episode Success Rate (Avg over Frames; from system_succ_hist)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3); plt.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    out_succ = os.path.join(out_dir, f"per_episode_success_rate_U{M_total}S{num_slots}.pdf")
    plt.savefig(out_succ, dpi=600, bbox_inches="tight"); plt.close()

    print("[PLOT] Saved:",
          f"\n  {out_reward}" if out_reward else "\n  (reward plot skipped)",
          f"\n  {out_succ}")

    return {
        "episode_index": ep_idx,
        "per_episode_avg_reward": ep_avg_reward,                   # may contain NaN if rewards missing
        "per_episode_avg_reward_normalized": ep_avg_reward_norm,   # None if reward plot skipped
        "per_episode_success_rate": ep_success,                    # derived from state hist
    }


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
    rolling_window=None,                # e.g., 1000 for a smoother auxiliary curve
    # --- new knobs ---
    include_per_slot_in_main=True,      # True -> keep thin per-slot mean in the main figure
    save_avg_only=True,                 # True -> also save a second "averages-only" figure
    avg_only_pdf="system_aoi_time_avg_only.pdf",
    avg_ylim_clip=(1, 99),              # y-axis clips by percentiles of average curves for better scale
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

    uids = sorted(data.keys())
    U = len(uids)
    T = num_slots * frames_per_episode
    T_total = T * num_episodes

    # --- A) Per-slot mean AoI across users (cross-sectional) ---
    sum_per_t, cnt_per_t = {}, {}
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

    # Optional rolling mean (centered simple moving average)
    # Optional rolling mean (centered simple moving average)
    roll_curve = None
    if rolling_window and rolling_window > 1:
        k = int(rolling_window)
        kernel = np.ones(k) / k
        num = np.convolve(ps, kernel, mode="same")
        den = np.convolve(w, kernel, mode="same")
        roll_curve = np.divide(num, np.maximum(den, 1e-12))

    # --- B) Mean of users' moving averages (built from exact per-user MAvgs) ---
    mean_of_user_mavgs = None
    if also_plot_mean_of_user_mavgs:
        # expects compute_user_moving_avgs to be available in scope
        user_series = compute_user_moving_avgs(data, num_slots, frames_per_episode, num_episodes)
        mtx = np.full((U, max_t + 1), np.nan, dtype=float)
        for ui, uid in enumerate(uids):
            t_u = user_series[uid]["t"]
            m_u = user_series[uid]["mavg"]
            if t_u.size == 0:
                continue
            row = np.full(max_t + 1, np.nan, dtype=float)
            row[t_u] = m_u
            # forward-fill each user's series
            last = np.nan
            for i in range(max_t + 1):
                if np.isfinite(row[i]):
                    last = row[i]
                else:
                    row[i] = last
            mtx[ui] = row
        mean_of_user_mavgs = np.nanmean(mtx, axis=0)

    # --- episode markers ---
    x = np.arange(max_t + 1)
    episode_boundaries = [i * T for i in range(0, num_episodes + 1)]

    # ------------------ Figure 1: main (optionally with per-slot) ------------------
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    if include_per_slot_in_main:
        ax.plot(x, per_slot_mean, linewidth=0.9, alpha=0.35, label="Per-slot mean AoI (across users)")
    ax.plot(x, system_running_mean, linewidth=1.6, label="Running mean AoI (system)")
    if roll_curve is not None:
        ax.plot(x, roll_curve, linestyle="--", linewidth=1.2, label=f"Rolling mean (w={rolling_window})")
    if mean_of_user_mavgs is not None:
        ax.plot(x, mean_of_user_mavgs, linewidth=1.2, label="Mean of users' moving avgs")

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

    # ------------------ Figure 2: averages-only (no per-slot) ------------------
    if save_avg_only:
        fig2, ax2 = plt.subplots(figsize=(7.8, 3.2))
        lines = []
        labels = []

        l = ax2.plot(x, system_running_mean, linewidth=1.8, label="Running mean AoI (system)")[0]
        lines.append(l); labels.append(l.get_label())

        if roll_curve is not None:
            l = ax2.plot(x, roll_curve, linestyle="--", linewidth=1.4,
                         label=f"Rolling mean (w={rolling_window})")[0]
            lines.append(l); labels.append(l.get_label())

        if mean_of_user_mavgs is not None:
            l = ax2.plot(x, mean_of_user_mavgs, linewidth=1.4,
                         label="Mean of users' moving avgs")[0]
            lines.append(l); labels.append(l.get_label())

        for eb in episode_boundaries:
            if eb <= max_t:
                ax2.axvline(eb, color="0.9", linewidth=0.6, zorder=0)

        ax2.set_xlim(0, max_t)
        ax2.set_xlabel("Slot Index")
        ax2.set_ylabel("AoI")
        ax2.set_title("System AoI — Averages Only")
        ax2.grid(True, alpha=0.3)
        ax2.legend(frameon=False, fontsize=7, ncols=2)

        # --- smart y-limits based only on the averages drawn ---
        '''''''''
        yvals = []
        for ln in lines:
            y = ln.get_ydata()
            y = y[np.isfinite(y)]
            if y.size:
                yvals.append(y)
        if yvals:
            ystack = np.concatenate(yvals)
            lo = np.percentile(ystack, avg_ylim_clip[0])
            hi = np.percentile(ystack, avg_ylim_clip[1])
            if lo == hi:
                pad = 0.05 * (abs(hi) + 1.0)
                lo, hi = hi - pad, hi + pad
            # small margin
            margin = 0.05 * (hi - lo if hi > lo else 1.0)
            ax2.set_ylim(lo - margin, hi + margin)
        '''

        fig2.tight_layout()
        out_path2 = os.path.join(out_dir, avg_only_pdf)
        fig2.savefig(out_path2, dpi=600, format="pdf", bbox_inches="tight")
        plt.close(fig2)
        print(f"[PLOT] System AoI (averages-only) saved → {out_path2}")


import os, sqlite3, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl, math

def open_db(run_dir, filename="slotwise_data.sqlite"):
    path = os.path.join(run_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return sqlite3.connect(path)

def list_uids(conn):
    cur = conn.cursor()
    return [int(r[0]) for r in cur.execute("SELECT DISTINCT uid FROM logs ORDER BY uid")]

def iter_user_rows(conn, uid):
    """Yield rows for one user, sorted by (ep,frame,slot)."""
    cur = conn.cursor()
    q = ("SELECT ep,frame,slot,aoi FROM logs "
         "WHERE uid=? ORDER BY ep,frame,slot")
    for ep,fr,sl,aoi in cur.execute(q, (int(uid),)):
        yield int(ep), int(fr), int(sl), float(aoi)

'''''''''
def plot_moving_avg_aoi_per_user_sqlite(
    conn, num_slots, frames_per_episode, num_episodes,
    out_dir, out_pdf="All_Users_MovingAvgAoI.pdf",
    episode_tick=5, y_min=300.0
):
    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams.update({
        "font.family":"serif","mathtext.fontset":"stix","axes.unicode_minus":False,
        "pdf.use14corefonts":True,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
        "axes.linewidth":0.8,"lines.linewidth":1.1,"grid.linewidth":0.5,
        "xtick.major.width":0.6,"ytick.major.width":0.6,
    })

    uids = list_uids(conn)
    ncols, nrows = 3, max(1, math.ceil(len(uids)/3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3*ncols, 2.1*nrows), squeeze=False)
    axes = axes.ravel()

    T_ep = num_slots*frames_per_episode
    Ttot = T_ep*num_episodes
    episode_end_slots = [(e*T_ep)-1 for e in range(1, num_episodes+1)]
    xticks = episode_end_slots[::max(1, episode_tick)]

    for ii, uid in enumerate(uids):
        ax = axes[ii]
        cum = 0.0; idx = 0
        t_list = []
        avg_list = []

        # stream rows for this user
        for ep, fr, sl, aoi in iter_user_rows(conn, uid):
            if 1 <= ep <= num_episodes:
                g = (ep-1)*T_ep + fr*num_slots + sl
                if 0 <= g < Ttot:
                    cum += aoi; idx += 1
                    t_list.append(g)
                    avg_list.append(cum/idx)

        if not t_list:
            ax.set_title(f"U{uid} (no data)", fontsize=8); ax.axis("off"); continue

        ax.plot(t_list, avg_list, linestyle="-", linewidth=1.3, label=f"U{uid}")

        # green hollow markers at each episode end where we have a sample
        ends_present = set(t_list).intersection(episode_end_slots)
        if ends_present:
            mark_x = sorted(list(ends_present))
            # map x -> y by finding its index once (keep O(n) by streaming)
            pos = {t:i for i,t in enumerate(t_list)}
            mark_y = [avg_list[pos[x]] for x in mark_x]
            ax.plot(mark_x, mark_y, "o", markerfacecolor="none", markeredgecolor="green",
                    markersize=5, linestyle="None")

        ax.set_ylim(bottom=y_min)
        ax.set_xticks(xticks)
        ax.set_xlim([0, Ttot])
        ax.set_title(f"U{uid}", fontsize=8)
        ax.set_xlabel("Slot Index"); ax.set_ylabel("Average AoI")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False, fontsize=7)

    for j in range(len(uids), nrows*ncols):
        axes[j].axis("off")

    fig.tight_layout(pad=0.6)
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")
'''

import os, math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_moving_avg_aoi_per_user_sqlite(
    conn, num_slots, frames_per_episode, num_episodes,
    out_dir, out_pdf="All_Users_MovingAvgAoI.pdf",
    episode_tick=5, y_min=None   # let y_min auto unless you really want a floor
):
    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams.update({
        "font.family":"serif","mathtext.fontset":"stix","axes.unicode_minus":False,
        "pdf.use14corefonts":True,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
        "axes.linewidth":0.8,"lines.linewidth":1.1,"grid.linewidth":0.5,
        "xtick.major.width":0.6,"ytick.major.width":0.6,
    })

    def list_uids(conn):
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT uid FROM logs ORDER BY uid")
        return [row[0] for row in cur.fetchall()]

    # **IMPORTANT**: make sure we fetch in strict time order
    def iter_user_rows_ordered(conn, uid):
        cur = conn.cursor()
        q = ("SELECT ep, frame, slot, aoi FROM logs "
             "WHERE uid=? ORDER BY ep ASC, frame ASC, slot ASC")
        for ep, fr, sl, aoi in cur.execute(q, (int(uid),)):
            yield int(ep), int(fr), int(sl), float(aoi)

    uids = list_uids(conn)
    T_ep  = int(num_slots) * int(frames_per_episode)
    Ttot  = T_ep * int(num_episodes)
    episode_end_slots = [(e*T_ep)-1 for e in range(1, num_episodes+1)]
    xticks = episode_end_slots[::max(1, episode_tick)]

    ncols, nrows = 3, max(1, math.ceil(len(uids)/3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3*ncols, 2.1*nrows), squeeze=False)
    axes = axes.ravel()

    for ii, uid in enumerate(uids):
        ax = axes[ii]

        # ---- Gather rows then sort by global slot index g ----
        rows = []
        for ep, fr, sl, aoi in iter_user_rows_ordered(conn, uid):
            if 1 <= ep <= num_episodes:
                if 0 <= sl < num_slots and 0 <= fr < frames_per_episode:
                    g = (ep-1)*T_ep + fr*num_slots + sl
                    if 0 <= g < Ttot:
                        rows.append((g, aoi))
        if not rows:
            ax.set_title(f"U{uid} (no data)", fontsize=8); ax.axis("off"); continue

        rows.sort(key=lambda x: x[0])  # sort by global slot index
        t_list = [g for g,_ in rows]
        aoi_seq = np.array([a for _,a in rows], dtype=float)

        # ---- Cumulative average over *sorted* time ----
        cum = np.cumsum(aoi_seq)
        idx = np.arange(1, len(aoi_seq)+1, dtype=float)
        avg_list = (cum / idx).tolist()

        # ---- Plot ----
        ax.plot(t_list, avg_list, linestyle="-", linewidth=1.3, label=f"U{uid}")

        # Mark each episode end if present
        ends_present = sorted(set(t_list).intersection(episode_end_slots))
        if ends_present:
            pos = {t:i for i,t in enumerate(t_list)}
            mark_y = [avg_list[pos[x]] for x in ends_present]
            ax.plot(ends_present, mark_y, "o", markerfacecolor="none", markeredgecolor="green",
                    markersize=4.5, linestyle="None")

        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        ax.set_xticks(xticks)
        ax.set_xlim([0, Ttot])
        ax.set_title(f"U{uid}", fontsize=8)
        ax.set_xlabel("Global Slot Index"); ax.set_ylabel("Average AoI")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=False, fontsize=7)

    for j in range(len(uids), nrows*ncols):
        axes[j].axis("off")

    fig.tight_layout(pad=0.6)
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")


def plot_system_avg_aoi_sqlite_ma(
    conn, num_slots, frames_per_episode, num_episodes,
    out_dir, out_pdf="system_avg_aoi_ma.pdf",
    ma_window=10   # moving-average window size
):
    """
    Plot per-episode system average AoI (from SQLite logs) with optional moving-average smoothing.
    """
    os.makedirs(out_dir, exist_ok=True)

    def list_uids(conn):
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT uid FROM logs ORDER BY uid")
        return [row[0] for row in cur.fetchall()]

    uids = list_uids(conn)
    T_ep = num_slots * frames_per_episode

    system_avg_per_ep = []
    for ep in range(1, num_episodes + 1):
        per_user_final = []
        for uid in uids:
            cum = 0.0
            k = 0
            cur = conn.cursor()
            q = (
                "SELECT frame, slot, aoi FROM logs "
                "WHERE uid=? AND ep=? ORDER BY frame, slot"
            )
            for fr, sl, aoi in cur.execute(q, (int(uid), int(ep))):
                cum += float(aoi)
                k += 1
            if k > 0:
                per_user_final.append(cum / k)
        if per_user_final:
            system_avg_per_ep.append(sum(per_user_final) / len(per_user_final))
        else:
            system_avg_per_ep.append(np.nan)

    # compute moving average (ignoring NaNs)
    def moving_average(x, w):
        x = np.asarray(x, dtype=float)
        mask = np.isfinite(x)
        valid = np.where(mask, x, 0)
        counts = np.convolve(mask.astype(float), np.ones(w), mode='valid')
        sums = np.convolve(valid, np.ones(w), mode='valid')
        with np.errstate(invalid='ignore'):
            ma = np.divide(sums, counts, where=counts > 0)
        return ma

    ma_values = moving_average(system_avg_per_ep, ma_window)
    ma_eps = np.arange(ma_window, ma_window + len(ma_values))

    # plot
    plt.figure(figsize=(6.6, 3.2))
    plt.plot(range(1, num_episodes + 1), system_avg_per_ep,
             alpha=0.3, label="Raw", linestyle='-', linewidth=1.0)
    plt.plot(ma_eps, ma_values,
             label=f"Moving Avg (k={ma_window})", linewidth=1.8)
    plt.title("System Average AoI per Episode")
    plt.xlabel("Episode")
    plt.ylabel("System Avg AoI")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()

    out_path = os.path.join(out_dir, out_pdf)
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved → {out_path}")

    return system_avg_per_ep, ma_values

def plot_system_avg_aoi_sqlite(
    conn, num_slots, frames_per_episode, num_episodes,
    out_dir, out_pdf="system_avg_aoi.pdf"
):
    os.makedirs(out_dir, exist_ok=True)
    import matplotlib.pyplot as plt

    uids = list_uids(conn)
    T_ep = num_slots*frames_per_episode

    system_avg_per_ep = []
    for ep in range(1, num_episodes+1):
        per_user_final = []
        for uid in uids:
            cum = 0.0; k = 0
            # stream only this episode for this user
            cur = conn.cursor()
            q = ("SELECT frame,slot,aoi FROM logs "
                 "WHERE uid=? AND ep=? ORDER BY frame,slot")
            for fr,sl,aoi in cur.execute(q, (int(uid), int(ep))):
                cum += float(aoi); k += 1
            if k > 0:
                per_user_final.append(cum/k)   # final average at end of ep (cumulative)
        if per_user_final:
            system_avg_per_ep.append(sum(per_user_final)/len(per_user_final))
        else:
            system_avg_per_ep.append(np.nan)

    # plot
    plt.figure()
    plt.plot(range(1, num_episodes+1), system_avg_per_ep, marker='s', linestyle='--')
    plt.title("System Average AoI per Episode")
    plt.xlabel("Episode"); plt.ylabel("System Avg AoI"); plt.grid(True)
    plt.tight_layout()
    out_path = os.path.join(out_dir, out_pdf)
    plt.savefig(out_path); plt.close()
    print(f"[PLOT] Saved → {out_path}")
    return system_avg_per_ep

def plot_system_avg_aoi_timewise_strict_sqlite(
    conn,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir,
    out_pdf="system_aoi_time_avg.pdf",
    rolling_window=None,            # e.g., 1000
    include_per_slot_in_main=True,
    save_avg_only=True,
    avg_only_pdf="system_aoi_time_avg_only.pdf"
):
    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams.update({
        "font.family":"serif","mathtext.fontset":"stix","axes.unicode_minus":False,
        "pdf.use14corefonts":True,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
        "axes.linewidth":0.8,"lines.linewidth":1.1,"grid.linewidth":0.5,
        "xtick.major.width":0.6,"ytick.major.width":0.6,
    })

    T_ep = num_slots*frames_per_episode
    Ttot = T_ep*num_episodes

    # stream whole table ordered by time; accumulate per-slot mean across users
    sum_per_t = np.zeros(Ttot, dtype=float)
    cnt_per_t = np.zeros(Ttot, dtype=int)

    cur = conn.cursor()
    # ordering by (ep,frame,slot,uid) ensures per-slot aggregation is correct
    q = ("SELECT ep,frame,slot,aoi FROM logs ORDER BY ep,frame,slot,uid")
    for ep, fr, sl, aoi in cur.execute(q):
        ep = int(ep); fr = int(fr); sl = int(sl)
        if 1 <= ep <= num_episodes:
            g = (ep-1)*T_ep + fr*num_slots + sl
            if 0 <= g < Ttot:
                sum_per_t[g] += float(aoi); cnt_per_t[g] += 1

    # compute per-slot mean where we have any sample; forward-fill gaps for stability
    per_slot_mean = np.full(Ttot, np.nan, dtype=float)
    mask = cnt_per_t > 0
    per_slot_mean[mask] = sum_per_t[mask] / cnt_per_t[mask]
    last = np.nan
    for i in range(Ttot):
        if np.isfinite(per_slot_mean[i]):
            last = per_slot_mean[i]
        else:
            per_slot_mean[i] = last

    # running mean over time
    valid = np.isfinite(per_slot_mean)
    ps = np.where(valid, per_slot_mean, 0.0)
    w  = np.where(valid, 1.0, 0.0)
    csum = np.cumsum(ps); wsum = np.cumsum(w)
    running_mean = np.divide(csum, np.maximum(wsum, 1e-12))

    # optional rolling mean (same streaming arrays)
    roll_curve = None
    if rolling_window and rolling_window > 1:
        k = int(rolling_window)
        kernel = np.ones(k)/k
        num = np.convolve(ps, kernel, mode="same")
        den = np.convolve(w,  kernel, mode="same")
        roll_curve = np.divide(num, np.maximum(den, 1e-12))

    # episode markers
    episode_bounds = [i*T_ep for i in range(0, num_episodes+1)]
    x = np.arange(Ttot)

    # Figure 1
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    if include_per_slot_in_main:
        ax.plot(x, per_slot_mean, linewidth=0.9, alpha=0.35, label="Per-slot mean AoI")
    ax.plot(x, running_mean, linewidth=1.6, label="Running mean AoI (system)")
    if roll_curve is not None:
        ax.plot(x, roll_curve, linestyle="--", linewidth=1.2, label=f"Rolling mean (w={rolling_window})")
    for eb in episode_bounds:
        ax.axvline(eb, color="0.88", linewidth=0.7, zorder=0)
    ax.set_xlim(0, Ttot-1)
    ax.set_xlabel("Slot Index"); ax.set_ylabel("AoI"); ax.set_title("System Average AoI Over Time")
    ax.grid(True, alpha=0.3); ax.legend(frameon=False, fontsize=7, ncols=2)
    fig.tight_layout()
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight"); plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")

    # Figure 2 (averages only)
    if save_avg_only:
        fig2, ax2 = plt.subplots(figsize=(7.8, 3.2))
        ax2.plot(x, running_mean, linewidth=1.8, label="Running mean AoI (system)")
        if roll_curve is not None:
            ax2.plot(x, roll_curve, linestyle="--", linewidth=1.4, label=f"Rolling mean (w={rolling_window})")
        for eb in episode_bounds:
            ax2.axvline(eb, color="0.9", linewidth=0.6, zorder=0)
        ax2.set_xlim(0, Ttot-1)
        ax2.set_xlabel("Slot Index"); ax2.set_ylabel("AoI"); ax2.set_title("System AoI — Averages Only")
        ax2.grid(True, alpha=0.3); ax2.legend(frameon=False, fontsize=7, ncols=2)
        fig2.tight_layout()
        out_path2 = os.path.join(out_dir, "system_aoi_time_avg_only.pdf" if avg_only_pdf is None else avg_only_pdf)
        fig2.savefig(out_path2, dpi=600, format="pdf", bbox_inches="tight"); plt.close(fig2)
        print(f"[PLOT] Saved → {out_path2}")

'''''''''
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
    '''

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_episode_end_avg_and_variance_sqlite(
    conn,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir,
    out_pdf="system_episode_end_avg_var.pdf",
    ddof=0,        # 0: population variance, 1: sample
    sigma_shading=True  # whether to add ±1σ shading around system_end_avg
):
    """
    For each episode:
      - Compute users’ running-average AoI over slots in that episode.
      - Record each user’s last running-average value and variance across time.
      - Aggregate across users:
          (1) system_end_avg  = mean of users’ final averages.
          (2) system_end_var_intra = mean of users’ variances (temporal stability).
          (3) system_end_var_inter = variance across users’ final averages (dispersion).
      - Plot:
          - Blue line = system_end_avg (AoI convergence)
          - Red dashed = mean per-user variance (intra)
          - Orange dash-dot = inter-user variance
          - Optional orange shaded band = ±1σ across users at each episode end.
    """
    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams.update({
        "font.family":"serif","mathtext.fontset":"stix","axes.unicode_minus":False,
        "pdf.use14corefonts":True,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
        "axes.linewidth":0.8,"lines.linewidth":1.1,"grid.linewidth":0.5,
        "xtick.major.width":0.6,"ytick.major.width":0.6,
    })

    T_ep = int(num_slots) * int(frames_per_episode)
    Ttot = T_ep * int(num_episodes)

    # Output arrays
    system_end_avg = np.full(num_episodes, np.nan, dtype=float)
    system_end_var_intra = np.full(num_episodes, np.nan, dtype=float)
    system_end_var_inter = np.full(num_episodes, np.nan, dtype=float)
    x_pos = np.array([(e * T_ep) - 1 for e in range(1, num_episodes + 1)], dtype=int)

    cur = conn.cursor()

    for e in range(1, num_episodes + 1):
        cum_sum, n_seen, mean_m, M2_m, last_m = {}, {}, {}, {}, {}

        q = ("SELECT uid, aoi FROM logs "
             "WHERE ep=? ORDER BY frame ASC, slot ASC, uid ASC")
        for uid, aoi in cur.execute(q, (e,)):
            uid = int(uid); aoi = float(aoi)
            if uid not in n_seen:
                n_seen[uid]  = 0
                cum_sum[uid] = 0.0
                mean_m[uid]  = 0.0
                M2_m[uid]    = 0.0

            n_seen[uid]  += 1
            cum_sum[uid] += aoi
            m_t = cum_sum[uid] / n_seen[uid]
            last_m[uid] = m_t

            # Welford’s algorithm for variance of m_t sequence
            n = n_seen[uid]
            if n == 1:
                mean_m[uid] = m_t
                M2_m[uid]   = 0.0
            else:
                delta = m_t - mean_m[uid]
                mean_m[uid] += delta / n
                M2_m[uid]   += delta * (m_t - mean_m[uid])

        # Aggregate across users
        end_vals, var_vals = [], []
        for uid in last_m.keys():
            end_vals.append(last_m[uid])
            n = n_seen[uid]
            if n - ddof > 0:
                var_vals.append(M2_m[uid] / (n - ddof))
            else:
                var_vals.append(np.nan)

        if end_vals:
            end_vals = np.array(end_vals, dtype=float)
            system_end_avg[e - 1] = np.nanmean(end_vals)
            system_end_var_inter[e - 1] = np.nanvar(end_vals, ddof=ddof)
        if var_vals:
            system_end_var_intra[e - 1] = np.nanmean(var_vals)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    ax.plot(x_pos, system_end_avg, marker="o", linestyle="-", linewidth=1.4,
            markersize=4.0, color="tab:blue", label="System episode-end avg AoI")

    # ±1σ shading from inter-user variance
    if sigma_shading and np.isfinite(system_end_var_inter).any():
        std_band = np.sqrt(system_end_var_inter)
        upper = system_end_avg + std_band
        lower = system_end_avg - std_band
        ax.fill_between(x_pos, lower, upper, color="tab:orange", alpha=0.15,
                        label="±1σ (inter-user spread)")

    # vertical lines for episodes
    for xb in range(0, Ttot + 1, T_ep):
        ax.axvline(xb, color="0.9", linewidth=0.6, zorder=0)

    ax.set_xlim(-0.5, max(x_pos) + 0.5)
    ax.set_xlabel("Global slot index (episode ends)")
    ax.set_ylabel("AoI")
    ax.set_title("System Average AoI and Variances per Episode End")
    ax.grid(True, alpha=0.3)

    # Right axis for variances
    ax_r = ax.twinx()
    finite_intra = np.isfinite(system_end_var_intra)
    finite_inter = np.isfinite(system_end_var_inter)

    lines, labels = [], []
    if finite_intra.any():
        l = ax_r.plot(x_pos[finite_intra], np.sqrt(system_end_var_intra[finite_intra]),
                      marker="s", linestyle="--", linewidth=1.0, markersize=4.0,
                      color="tab:red", label="Mean user variance (intra)")[
            0
        ]
        lines.append(l); labels.append(l.get_label())

    if finite_inter.any():
        l = ax_r.plot(x_pos[finite_inter], np.sqrt(system_end_var_inter[finite_inter]),
                      marker="^", linestyle="-.", linewidth=1.0, markersize=4.0,
                      color="tab:orange", label="Variance across users (inter)")[
            0
        ]
        lines.append(l); labels.append(l.get_label())

    if lines:
        l1, lab1 = ax.get_legend_handles_labels()
        ax.legend(l1 + lines, lab1 + labels, frameon=False, fontsize=7, loc="best")
        ax_r.set_ylabel("Variance")

    fig.tight_layout()
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")

    return {
        "episode_end_x": x_pos,
        "system_end_avg": system_end_avg,
        "system_end_var_intra": system_end_var_intra,
        "system_end_var_inter": system_end_var_inter,
    }

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


def plot_system_avg_aoi_timewise_strict_sqlite_var(
    conn,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir,
    out_pdf="system_aoi_time_avg.pdf",
    rolling_window=None,            # e.g., 1000
    include_per_slot_in_main=True,
    save_avg_only=True,
    avg_only_pdf="system_aoi_time_avg_only.pdf"
):
    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.use14corefonts": True,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.1,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    T_ep = int(num_slots) * int(frames_per_episode)
    Ttot = T_ep * int(num_episodes)

    # ---------- Aggregate per-slot mean AoI ----------
    sum_per_t = np.zeros(Ttot, dtype=float)
    cnt_per_t = np.zeros(Ttot, dtype=int)

    cur = conn.cursor()
    q = "SELECT ep,frame,slot,aoi FROM logs ORDER BY ep,frame,slot,uid"
    for ep, fr, sl, aoi in cur.execute(q):
        ep = int(ep)
        fr = int(fr)
        sl = int(sl)
        if 1 <= ep <= num_episodes:
            g = (ep - 1) * T_ep + fr * num_slots + sl
            if 0 <= g < Ttot:
                sum_per_t[g] += float(aoi)
                cnt_per_t[g] += 1

    per_slot_mean = np.full(Ttot, np.nan)
    mask = cnt_per_t > 0
    per_slot_mean[mask] = sum_per_t[mask] / cnt_per_t[mask]

    # forward-fill missing slots
    last = np.nan
    for i in range(Ttot):
        if np.isfinite(per_slot_mean[i]):
            last = per_slot_mean[i]
        else:
            per_slot_mean[i] = last

    # ---------- Running mean AoI ----------
    valid = np.isfinite(per_slot_mean)
    ps = np.where(valid, per_slot_mean, 0.0)
    w = np.where(valid, 1.0, 0.0)
    csum = np.cumsum(ps)
    wsum = np.cumsum(w)
    running_mean = np.divide(csum, np.maximum(wsum, 1e-12))

    # ---------- Rolling mean ----------
    roll_curve = None
    if rolling_window and rolling_window > 1:
        k = int(rolling_window)
        kernel = np.ones(k) / k
        num = np.convolve(ps, kernel, mode="same")
        den = np.convolve(w, kernel, mode="same")
        roll_curve = np.divide(num, np.maximum(den, 1e-12))

    # ---------- Variance of running mean per episode ----------
    ep_var = np.full(num_episodes, np.nan)
    ep_pos = np.zeros(num_episodes, dtype=int)

    for e in range(1, num_episodes + 1):
        t0 = (e - 1) * T_ep
        t1 = e * T_ep
        seg = running_mean[t0:t1]
        seg = seg[np.isfinite(seg)]
        if seg.size > 1:
            ep_var[e - 1] = np.var(seg, ddof=1)  # sample variance
        ep_pos[e - 1] = t1 - 1  # plot position at episode end

    # ---------- Plot ----------
    x = np.arange(Ttot)
    episode_bounds = [i * T_ep for i in range(0, num_episodes + 1)]
    var_ok = np.isfinite(ep_var)

    # ========== Figure 1: main ==========
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    if include_per_slot_in_main:
        ax.plot(x, per_slot_mean, linewidth=0.9, alpha=0.35, label="Per-slot mean AoI")
    ax.plot(x, running_mean, linewidth=1.6, label="Running mean AoI (system)")
    if roll_curve is not None:
        ax.plot(x, roll_curve, linestyle="--", linewidth=1.2,
                label=f"Rolling mean (w={rolling_window})")

    for eb in episode_bounds:
        ax.axvline(eb, color="0.88", linewidth=0.7, zorder=0)

    ax.set_xlim(0, Ttot - 1)
    ax.set_xlabel("Slot Index")
    ax.set_ylabel("AoI")
    ax.set_title("System Average AoI Over Time")
    ax.grid(True, alpha=0.3)

    # variance on secondary y-axis
    ax_r = ax.twinx()
    if var_ok.any():
        ax_r.plot(ep_pos[var_ok], ep_var[var_ok],
                  marker="o", markersize=3.5, linewidth=0.9, alpha=0.85, color="tab:red",
                  label="Variance of running mean AoI (per episode)")
        ax_r.set_ylabel("Variance (per episode)", rotation=270, labelpad=12)

        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax_r.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=7, ncols=2, loc="best")
    else:
        ax.legend(frameon=False, fontsize=7, ncols=2, loc="best")

    fig.tight_layout()
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")

    # ========== Figure 2: averages only ==========
    if save_avg_only:
        fig2, axL = plt.subplots(figsize=(7.8, 3.2))
        axL.plot(x, running_mean, linewidth=1.8, label="Running mean AoI (system)")
        if roll_curve is not None:
            axL.plot(x, roll_curve, linestyle="--", linewidth=1.4,
                     label=f"Rolling mean (w={rolling_window})")
        for eb in episode_bounds:
            axL.axvline(eb, color="0.9", linewidth=0.6, zorder=0)

        axL.set_xlim(0, Ttot - 1)
        axL.set_xlabel("Slot Index")
        axL.set_ylabel("AoI")
        axL.set_title("System AoI — Averages Only")
        axL.grid(True, alpha=0.3)

        axR = axL.twinx()
        if var_ok.any():
            axR.plot(ep_pos[var_ok], ep_var[var_ok],
                     marker="o", markersize=3.5, linewidth=0.9, alpha=0.85, color="tab:red",
                     label="Variance of running mean AoI (per episode)")
            axR.set_ylabel("Variance (per episode)", rotation=270, labelpad=12)
            l1, lab1 = axL.get_legend_handles_labels()
            l2, lab2 = axR.get_legend_handles_labels()
            axL.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=7, ncols=2, loc="best")
        else:
            axL.legend(frameon=False, fontsize=7, ncols=2, loc="best")

        fig2.tight_layout()
        out_path2 = os.path.join(out_dir, avg_only_pdf)
        fig2.savefig(out_path2, dpi=600, format="pdf", bbox_inches="tight")
        plt.close(fig2)
        print(f"[PLOT] Saved → {out_path2}")

    return dict(
        per_slot_mean=per_slot_mean,
        running_mean=running_mean,
        rolling_mean=roll_curve,
        running_mean_var=ep_var,
        running_mean_var_x=ep_pos
    )

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_system_avg_aoi_timewise_strict_sqlite_varuser(
    conn,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir,
    out_pdf="system_aoi_time_avg.pdf",
    rolling_window=None,            # e.g., 1000
    include_per_slot_in_main=True,
    save_avg_only=True,
    avg_only_pdf="system_aoi_time_avg_only.pdf",
    ep_end_var_ddof=0               # 0 = population variance, 1 = sample variance
):
    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams.update({
        "font.family":"serif","mathtext.fontset":"stix","axes.unicode_minus":False,
        "pdf.use14corefonts":True,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,
        "axes.linewidth":0.8,"lines.linewidth":1.1,"grid.linewidth":0.5,
        "xtick.major.width":0.6,"ytick.major.width":0.6,
    })

    T_ep  = int(num_slots) * int(frames_per_episode)
    Ttot  = T_ep * int(num_episodes)

    # ---- Per-slot mean AoI across users (streamed) ----
    sum_per_t = np.zeros(Ttot, dtype=float)
    cnt_per_t = np.zeros(Ttot, dtype=int)

    cur = conn.cursor()
    q = "SELECT ep,frame,slot,aoi FROM logs ORDER BY ep,frame,slot,uid"
    for ep, fr, sl, aoi in cur.execute(q):
        ep = int(ep); fr = int(fr); sl = int(sl)
        if 1 <= ep <= num_episodes:
            g = (ep - 1) * T_ep + fr * num_slots + sl
            if 0 <= g < Ttot:
                sum_per_t[g] += float(aoi)
                cnt_per_t[g] += 1

    per_slot_mean = np.full(Ttot, np.nan, dtype=float)
    mask = cnt_per_t > 0
    per_slot_mean[mask] = sum_per_t[mask] / cnt_per_t[mask]
    # forward-fill
    last = np.nan
    for i in range(Ttot):
        if np.isfinite(per_slot_mean[i]):
            last = per_slot_mean[i]
        else:
            per_slot_mean[i] = last

    # ---- Running mean over time ----
    valid = np.isfinite(per_slot_mean)
    ps = np.where(valid, per_slot_mean, 0.0)
    w  = np.where(valid, 1.0, 0.0)
    csum = np.cumsum(ps); wsum = np.cumsum(w)
    running_mean = np.divide(csum, np.maximum(wsum, 1e-12))

    # ---- Optional rolling mean ----
    roll_curve = None
    if rolling_window and rolling_window > 1:
        k = int(rolling_window)
        kernel = np.ones(k) / k
        num = np.convolve(ps, kernel, mode="same")
        den = np.convolve(w,  kernel, mode="same")
        roll_curve = np.divide(num, np.maximum(den, 1e-12))

    # ---- NEW: Cross-sectional variance across users at episode end ----
    # At each episode end (frame = F-1, slot = S-1),
    # take the AoI across all users and compute variance.
    ep_end_frame = frames_per_episode - 1
    ep_end_slot  = num_slots - 1

    ep_end_var = np.full(num_episodes, np.nan, dtype=float)   # variance across users
    ep_end_mean = np.full(num_episodes, np.nan, dtype=float)  # (optional) mean across users at ep-end
    ep_pos = np.zeros(num_episodes, dtype=int)                # x-position to plot: episode end index

    q_end = ("SELECT aoi FROM logs WHERE ep=? AND frame=? AND slot=?")
    for e in range(1, num_episodes + 1):
        vals = [float(r[0]) for r in cur.execute(q_end, (e, ep_end_frame, ep_end_slot))]
        if len(vals) >= 2:
            ep_end_var[e - 1] = np.var(vals, ddof=ep_end_var_ddof)
            ep_end_mean[e - 1] = float(np.mean(vals))
        elif len(vals) == 1:
            ep_end_var[e - 1] = 0.0
            ep_end_mean[e - 1] = vals[0]
        ep_pos[e - 1] = e * T_ep - 1

    # ---- Plotting ----
    x = np.arange(Ttot)
    episode_bounds = [i * T_ep for i in range(0, num_episodes + 1)]
    var_ok = np.isfinite(ep_end_var)

    # ===================== Figure 1: main =====================
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    if include_per_slot_in_main:
        ax.plot(x, per_slot_mean, linewidth=0.9, alpha=0.35, label="Per-slot mean AoI")
    ax.plot(x, running_mean, linewidth=1.6, label="Running mean AoI (system)")
    if roll_curve is not None:
        ax.plot(x, roll_curve, linestyle="--", linewidth=1.2,
                label=f"Rolling mean (w={rolling_window})")

    for eb in episode_bounds:
        ax.axvline(eb, color="0.88", linewidth=0.7, zorder=0)

    ax.set_xlim(0, Ttot - 1)
    ax.set_xlabel("Slot Index")
    ax.set_ylabel("AoI")
    ax.set_title("System Average AoI Over Time")
    ax.grid(True, alpha=0.3)

    # Secondary Y-axis: variance across users at episode end
    ax_r = ax.twinx()
    if var_ok.any():
        ax_r.plot(ep_pos[var_ok], ep_end_var[var_ok],
                  marker="o", markersize=3.5, linewidth=0.9, alpha=0.9, color="tab:red",
                  label="Variance across users at episode end")
        ax_r.set_ylabel("Cross-sectional variance", rotation=270, labelpad=12)

        # unified legend
        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax_r.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=7, ncols=2, loc="best")
    else:
        ax.legend(frameon=False, fontsize=7, ncols=2, loc="best")

    fig.tight_layout()
    out_path = os.path.join(out_dir, out_pdf)
    fig.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")

    # ===================== Figure 2: averages only =====================
    if save_avg_only:
        fig2, axL = plt.subplots(figsize=(7.8, 3.2))
        axL.plot(x, running_mean, linewidth=1.8, label="Running mean AoI (system)")
        if roll_curve is not None:
            axL.plot(x, roll_curve, linestyle="--", linewidth=1.4,
                     label=f"Rolling mean (w={rolling_window})")
        for eb in episode_bounds:
            axL.axvline(eb, color="0.9", linewidth=0.6, zorder=0)

        axL.set_xlim(0, Ttot - 1)
        axL.set_xlabel("Slot Index")
        axL.set_ylabel("AoI")
        axL.set_title("System AoI — Averages Only")
        axL.grid(True, alpha=0.3)

        axR = axL.twinx()
        if var_ok.any():
            axR.plot(ep_pos[var_ok], ep_end_var[var_ok],
                     marker="o", markersize=3.5, linewidth=0.9, alpha=0.9, color="tab:red",
                     label="Variance across users at episode end")
            axR.set_ylabel("Cross-sectional variance", rotation=270, labelpad=12)
            l1, lab1 = axL.get_legend_handles_labels()
            l2, lab2 = axR.get_legend_handles_labels()
            axL.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=7, ncols=2, loc="best")
        else:
            axL.legend(frameon=False, fontsize=7, ncols=2, loc="best")

        fig2.tight_layout()
        out_path2 = os.path.join(out_dir, avg_only_pdf if avg_only_pdf else "system_aoi_time_avg_only.pdf")
        fig2.savefig(out_path2, dpi=600, format="pdf", bbox_inches="tight")
        plt.close(fig2)
        print(f"[PLOT] Saved → {out_path2}")

    return dict(
        per_slot_mean=per_slot_mean,
        running_mean=running_mean,
        rolling_mean=roll_curve,
        ep_end_var=ep_end_var,
        ep_end_var_x=ep_pos,
        ep_end_mean=ep_end_mean
    )

def make_run_dir(M_total, num_slots, num_episodes, gamma_th_db, tau):
    #stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"AoI_U{M_total}_S{num_slots}_EP{num_episodes}TH{gamma_th_db}_PolicyV22"
    out = os.path.join(name)
    os.makedirs(out, exist_ok=True)
    return out


# ------------------- Environment params (yours) -------------------
num_slots = 15
frames_per_episode = 200
num_episodes = 100
M_total = 75


gamma_th_db        = 0
gamma_th           = 10 ** (gamma_th_db / 10.0)
tau = 12
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


RUN_DIR = f"/Users/muhammadtauseefmushtaq/Documents/GitHub/AoI_PD_NOMA_PPO2/AOI_PPO_Users/AoI_U75_S15_EP100_RewardNS" #make_run_dir(M_total, num_slots, num_episodes, gamma_th_db, tau= tau)


# 4. >>> SAVE META FILE HERE <<<
#with open(os.path.join(RUN_DIR, "meta.json"), "w") as f:
 #   json.dump(run_meta, f, indent=2)

print(f"[SAVE] Run dir accessed: {RUN_DIR}")


#data = load_episode_telemetry(RUN_DIR, filename=f"slotwise_dataU{M_total}S{num_slots}.npy")
#plot_all_users_avg_aoi_combined(data, num_slots=num_slots, frames_per_episode=frames_per_episode, num_episodes = num_episodes ,out_dir="AoI_U6_S2_PPNR")
#plot_moving_avg_aoi_per_user(
#    data=data,
#    num_slots=num_slots,
#    frames_per_episode=frames_per_episode,
#    num_episodes=num_episodes,
#    out_dir=RUN_DIR,
#    out_pdf="All_Users_MovingAvgAoICluster2.pdf",
#    episode_tick=5
#)


#plot_system_avg_aoi(data, num_slots=num_slots, frames_per_episode=frames_per_episode,  out_dir=RUN_DIR, out_pdf="system_avg_aoicluster2.pdf")

#plot_slotwise_rewards(RUN_DIR, out_dir=RUN_DIR, window=1000)

#plot_time_averaged_system_aoi(data, num_slots, frames_per_episode, RUN_DIR, out_pdf="system_aoi_time_avg.pdf")


#plot_system_avg_aoi_timewise_strict(
 #   data=data,
  #  num_slots=num_slots,
   # frames_per_episode=frames_per_episode,
   # num_episodes=num_episodes,
   # out_dir=RUN_DIR,
   # out_pdf="system_aoi_time_avg.pdf",
   # also_plot_mean_of_user_mavgs=True,
   # rolling_window=1,   # optional
#)


#plot_system_avg_aoi_timewise(data, num_slots, frames_per_episode, RUN_DIR, out_pdf="system_aoi_time.pdf")

#plot_all_users_aoi(telemetry, num_slots, frames_per_episode,  out_pdf="AOI_All_Users.pdf", out_dir=RUN_DIR)
#plot_all_users_energy(telemetry, num_slots, frames_per_episode, out_pdf="Energy_All_Users.pdf", out_dir=RUN_DIR)


plot_state_action_pair_aoi(
    sar_log_dir=RUN_DIR,
    M_total=M_total,
    num_slots=num_slots,
    out_dir=os.path.join(RUN_DIR,"State Analysis"),
    n_bins_1d=25,
    n_bins_2d=30,
    clip_percentiles=(0.5, 99.5),
)
#plot_slotwise_rewards(RUN_DIR, out_dir=RUN_DIR, window=100)



plot_policy_analytics_modern(
    sar_log_dir=RUN_DIR,
    M_total=M_total,
    out_dir=os.path.join(RUN_DIR,"State Analysis"),
    num_slots=num_slots,
    pretty_labels=True

)



analyze_system_success_history_from_sar(
    sar_log_dir = RUN_DIR,
    M_total = M_total,
    num_slots = num_slots,
    frames_per_episode = frames_per_episode,
    core_state_len=12,          # number of non-history features at the start of each state
    hist_len=7,              # if None, infer as len(state) - core_state_len
    out_dir=os.path.join(RUN_DIR,"State Analysis"),
    ma_window=10                # moving-average over episodes for smoother curves
)



metrics = plot_episode_reward_curves_from_sar(
    sar_log_dir=RUN_DIR,         # wherever sar_logU{M_total}S{num_slots}.pkl lives
    M_total=M_total,
    num_slots=num_slots,
    frames_per_episode=frames_per_episode,  # <-- your value
    out_dir= os.path.join(RUN_DIR,"Reward Analysis"),
    normalize_mode="global",     # or "running" if you prefer online normalization
    ma_window=10,
)

'''''''''
conn = open_db(RUN_DIR, "slotwise_data.sqlite")

plot_system_avg_aoi_timewise_strict_sqlite_varuser(
    conn,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir =os.path.join(RUN_DIR,"AoI Analysis"),
    out_pdf="system_aoi_time_avgvaru.pdf",
    rolling_window=100,            # e.g., 1000
    include_per_slot_in_main=True,
    save_avg_only=True,
    avg_only_pdf="system_aoi_time_avg_only_varu.pdf",
    ep_end_var_ddof=0               # 0 = population variance, 1 = sample variance
)


plot_moving_avg_aoi_per_user_sqlite(
    conn, num_slots, frames_per_episode, num_episodes,
    out_dir=os.path.join(RUN_DIR,"AoI Analysis"),
    out_pdf="All_Users_MovingAvgAoI.pdf",
    episode_tick=5, y_min=1.0
)


plot_system_avg_aoi_sqlite(
    conn, num_slots, frames_per_episode, num_episodes,
    out_dir=os.path.join(RUN_DIR,"AoI Analysis"),
    out_pdf="system_avg_aoi.pdf"
)

plot_system_avg_aoi_timewise_strict_sqlite(
    conn, num_slots, frames_per_episode, num_episodes,
    out_dir=os.path.join(RUN_DIR,"AoI Analysis"),
    out_pdf="system_aoi_time_avg.pdf",
    rolling_window=300, include_per_slot_in_main=True
)

plot_system_avg_aoi_sqlite_ma(conn, num_slots, frames_per_episode, num_episodes,
                           out_dir=os.path.join(RUN_DIR,"AoI Analysis"), ma_window=20)


plot_system_avg_aoi_timewise_strict_sqlite_var(
    conn,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir =os.path.join(RUN_DIR,"AoI Analysis"),
    out_pdf="system_aoi_time_avg_var.pdf",
    rolling_window=100,            # e.g., 1000
    include_per_slot_in_main=True,
    save_avg_only=True,
    avg_only_pdf="system_aoi_time_avg_only_var.pdf"
)



plot_episode_end_avg_and_variance_sqlite(
    conn,
    num_slots,
    frames_per_episode,
    num_episodes,
    out_dir =os.path.join(RUN_DIR,"AoI Analysis"),
    out_pdf="system_episode_end_avg_var.pdf",
    ddof=0,        # 0: population variance, 1: sample
    sigma_shading=True  # whether to add ±1σ shading around system_end_avg
)

conn.close()
'''
'''''''''
plot_episode_reward_curves_from_sar(
    RUN_DIR,
    M_total,
    num_slots,
    frames_per_episode,
    out_dir=RUN_DIR,
    normalize_mode="global",   # "global" | "running"
    ma_window=10,
    fig3_mode="cumulative",    # "cumulative" | "per-episode"
    # --- convergence extras ---
    roll_window=10,            # rolling window (episodes) for mean/std/slope
    ewma_alpha=0.15,           # 0<alpha<=1 for EWMA; lower=more smoothing
    stability_tail=10          # show stability stats over last K episodes
)

'''

'''''''''
import os, sqlite3, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl

def episode_end_avg_series(conn, num_slots, frames_per_episode, num_episodes):
    """
    Returns array length = num_episodes.
    For each episode e:
      - build per-user running average over that episode's slots,
      - pick its LAST value (episode end) per user,
      - average across users -> system episode-end avg AoI for e.
    """
    series = np.full(num_episodes, np.nan, dtype=float)
    cur = conn.cursor()

    for e in range(1, num_episodes + 1):
        # Per-user cumulative running average within this episode
        cum_sum, n_seen, last_m = {}, {}, {}

        q = ("SELECT uid, aoi FROM logs "
             "WHERE ep=? ORDER BY frame ASC, slot ASC, uid ASC")
        for uid, aoi in cur.execute(q, (e,)):
            uid = int(uid); aoi = float(aoi)
            if uid not in n_seen:
                n_seen[uid]  = 0
                cum_sum[uid] = 0.0
            n_seen[uid]  += 1
            cum_sum[uid] += aoi
            last_m[uid] = cum_sum[uid] / n_seen[uid]  # running avg at this step

        if last_m:
            # system episode-end average = mean across users of their episode-end running avgs
            series[e - 1] = float(np.mean(list(last_m.values())))

    return series

def mean_of_episode_end_averages_for_run(run_dir, num_slots, frames_per_episode, num_episodes, db_name="slotwise_data.sqlite"):
    """
    One scalar for a run: mean over episodes of SYSTEM episode-end averages.
    """
    db_path = os.path.join(run_dir, db_name)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    ser = episode_end_avg_series(conn, num_slots, frames_per_episode, num_episodes)
    conn.close()
    # mean of episode-end averages (ignore NaNs if any)
    return float(np.nanmean(ser)), ser

def plot_mean_episode_end_avg_vs_users(run_dirs, user_counts, num_slots, frames_per_episode, num_episodes,
                                       out_pdf="mean_episode_end_avg_vs_users.pdf"):
    """
    For each run folder in run_dirs, compute the mean of episode-end system averages,
    then plot that single value vs M_total (user_counts) — IEEE two-column style.
    """
    import numpy as np
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
        # --- Larger readable labels for IEEE ---
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.8,
        "axes.titlesize": 9,
        # --- Lines and grid aesthetics ---
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.4,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    y_vals = []
    for M, run_dir in zip(user_counts, run_dirs):
        try:
            y, _ = mean_of_episode_end_averages_for_run(run_dir, num_slots, frames_per_episode, num_episodes)
            print(f"[OK] U={M}: mean episode-end avg AoI = {y:.3f}")
        except Exception as e:
            print(f"[WARN] U={M}: {e}")
            y = np.nan
        y_vals.append(y)

    # === Plot ===
    plt.figure(figsize=(IEEE_WIDTH, IEEE_HEIGHT))
    plt.plot(
        user_counts, y_vals,
        color="#6A0DAD",                      # purple line
        marker="o",
        markerfacecolor="none",               # hollow
        markeredgecolor="darkgreen",          # dark-green border
        markersize=5,
        linewidth=1.6,
        label="Proposed PPO"
    )

    plt.xlabel(r"Number of Users ($M$)")
    plt.ylabel("Network Average AoI")
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout(pad=0.3)
    plt.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"[SAVED] {out_pdf}")
    return user_counts, y_vals

num_slots = 15
frames_per_episode = 200
num_episodes = 100
user_counts = [60, 65, 70, 75]

run_dirs = [f"AoI_U{u}_S{num_slots}_EP{num_episodes}_RewardNS" for u in user_counts]

# Get the scalar per run + plot
xs, ys = plot_mean_episode_end_avg_vs_users(run_dirs, user_counts, num_slots, frames_per_episode, num_episodes,
                                            out_pdf="min_aoi_vs_users.pdf")  # rename as you like
'''