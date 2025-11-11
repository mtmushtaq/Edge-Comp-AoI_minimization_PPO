
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


def make_run_dir(M_total: int, num_slots: int, num_episodes: int) -> str:
    name = f"AoI_U{M_total}_S{num_slots}_LR3e32"
    out = os.path.join(name)
    os.makedirs(out, exist_ok=True)
    return out


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
