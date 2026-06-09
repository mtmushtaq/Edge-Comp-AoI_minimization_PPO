import os, sqlite3, hashlib, numpy as np

DB_NAME = "slotwise_data.sqlite"

def db_info(db_path, frames_per_episode, num_slots, num_eps_expected):
    if not os.path.exists(db_path):
        return {"exists": False}
    # hash bytes (fast enough; robust)
    h = hashlib.sha256()
    with open(db_path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    digest = h.hexdigest()

    con = sqlite3.connect(db_path); cur = con.cursor()
    try:
        rows_total = cur.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        eps_found  = cur.execute("SELECT MAX(ep) FROM logs").fetchone()[0] or 0
        # rows in ep=1
        rows_ep1   = cur.execute("SELECT COUNT(*) FROM logs WHERE ep=1").fetchone()[0]
        # sample
        sample = cur.execute("SELECT ep,frame,slot,uid,aoi FROM logs ORDER BY ep,frame,slot,uid LIMIT 5").fetchall()
    finally:
        con.close()

    return dict(exists=True, digest=digest, rows_total=rows_total,
                eps_found=eps_found, rows_ep1=rows_ep1,
                expected_rows_ep1=frames_per_episode*num_slots,
                sample=sample)

def episode_end_avg_series(db_path, num_slots, frames_per_episode, num_episodes):
    con = sqlite3.connect(db_path); cur = con.cursor()
    try:
        series = np.full(num_episodes, np.nan, dtype=float)
        for e in range(1, num_episodes + 1):
            cum_sum, n_seen, last_m = {}, {}, {}
            q = ("SELECT uid, aoi FROM logs "
                 "WHERE ep=? ORDER BY frame ASC, slot ASC, uid ASC")
            for uid, aoi in cur.execute(q, (e,)):
                uid = int(uid); aoi = float(aoi)
                if uid not in n_seen:
                    n_seen[uid] = 0; cum_sum[uid] = 0.0
                n_seen[uid] += 1; cum_sum[uid] += aoi
                last_m[uid] = cum_sum[uid] / n_seen[uid]
            if last_m:
                series[e-1] = float(np.mean(list(last_m.values())))
    finally:
        con.close()
    return series

def compare_two(db_a, db_b, num_slots, frames_per_episode, num_episodes, label_a, label_b):
    A = episode_end_avg_series(db_a, num_slots, frames_per_episode, num_episodes)
    B = episode_end_avg_series(db_b, num_slots, frames_per_episode, num_episodes)
    print(f"\n== Compare {label_a} vs {label_b} ==")
    print("means:", np.nanmean(A), np.nanmean(B))
    diff = A - B
    print("max|diff|:", np.nanmax(np.abs(diff)))
    print("first 5 A:", A[:5])
    print("first 5 B:", B[:5])

# ---- Set your exact folders for γ_th = -5 dB, M=60 ----
# Use the builders you finalized:
ppo_dir       = "AoI_U60_S15_GTH/AoI_U60_S15_EP50_GTHminus5"
thres_dir     = "AoI_U60_S15_Policy/AoI_U60_S15_EP50THminus5_PolicyV2"
greedy_dir    = "AoI_U60_S15_Greedy/AoI_U60_S15_EP50_Greedy_THminus5"
random_dir    = "AoI_U60_S15_THRNDM/AoI_U60_S15_EP50_THminus5RNDM"

fps, slots, eps = 200, 15, 50

for label, d in [("PPO", ppo_dir), ("Threshold", thres_dir), ("Greedy", greedy_dir), ("Random", random_dir)]:
    dbp = os.path.join(d, DB_NAME)
    info = db_info(dbp, fps, slots, eps)
    print(f"[{label}] path={dbp}\n  {info}")

# Example comparisons:
db_g = os.path.join(greedy_dir, DB_NAME)
db_t = os.path.join(thres_dir, DB_NAME)
compare_two(db_g, db_t, slots, fps, eps, "Greedy", "Threshold")
