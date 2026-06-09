# telemetry_flush.py
import os, gc, glob, tempfile
import numpy as np
from collections import defaultdict

# ---- public API ----
def new_buffer():
    """Create a fresh per-episode buffer: {uid -> [row, ...]}."""
    return defaultdict(list)

def append_slot(buf, uid, row_dict):
    """Append a lightweight row (dict with ep, frame, slot, aoi, ...)."""
    buf[int(uid)].append(row_dict)

def flush_episode(buf, run_dir, M_total, num_slots, ep):
    """
    Save ONE episode chunk to run_dir/chunks/slotwise_data_epXXXX_U{M}S{S}.npy
    and clear the in-RAM buffer.
    """
    if not buf:
        return
    chunks_dir = os.path.join(run_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    out_path = os.path.join(chunks_dir, f"slotwise_data_ep{int(ep):04d}_U{M_total}S{num_slots}.npy")
    _atomic_save_dict(out_path, dict(buf))
    buf.clear()
    gc.collect()
    print(f"[FLUSH] episode={ep} → {out_path}")

def finalize_run_and_write_final(buf, run_dir, M_total, num_slots,
                                 final_filename=None, keep_chunks=True):
    """
    Reassemble ALL episode chunks (+ anything leftover in buf) into the exact original file:
        final_filename (default: slotwise_dataU{M}S{S}.npy)
    The resulting object is: dict[uid:int] -> list[ row_dict, ... ].
    """
    if final_filename is None:
        final_filename = f"slotwise_dataU{M_total}S{num_slots}.npy"
    final_path = os.path.join(run_dir, final_filename)

    # 1) load all chunks
    chunks_dir = os.path.join(run_dir, "chunks")
    chunk_paths = sorted(glob.glob(os.path.join(
        chunks_dir, f"slotwise_data_ep*_U{M_total}S{num_slots}.npy")))
    merged = defaultdict(list)

    for cp in chunk_paths:
        try:
            d = np.load(cp, allow_pickle=True).item()
            # normalize uid keys to int
            for k, v in d.items():
                merged[int(k)].extend(v)
        except Exception as e:
            print(f"[WARN] failed to read chunk {cp}: {e}")

    # 2) add any leftovers still in RAM (last episode if not flushed)
    if buf:
        for k, v in buf.items():
            merged[int(k)].extend(v)
        buf.clear()

    # 3) write the final file (exact same path & format you’ve always used)
    _atomic_save_dict(final_path, dict(merged))
    print(f"[FINAL] wrote {final_path}  (uids={len(merged)})")

    # 4) optionally clean up chunks
    if not keep_chunks and chunk_paths:
        for cp in chunk_paths:
            try: os.remove(cp)
            except Exception: pass
        try:
            os.rmdir(chunks_dir)  # remove empty dir
        except Exception:
            pass

# ---- helpers ----
def _atomic_save_dict(path, payload_dict):
    """Atomic np.save of a Python dict (allow_pickle=True)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=os.path.dirname(path) or ".")
    os.close(fd)
    try:
        np.save(tmp, payload_dict, allow_pickle=True)
        if os.path.exists(path):
            os.remove(path)
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise
