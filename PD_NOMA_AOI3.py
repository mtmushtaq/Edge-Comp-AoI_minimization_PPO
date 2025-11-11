import os
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

EPS = 1e-8

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, n_slots):
        super().__init__()
        self.n_slots = n_slots
        self.f1 = nn.Linear(state_dim, 128)
        self.f2 = nn.Linear(128, 128)
        self.pi = nn.Linear(128, n_slots)
        self.v = nn.Linear(128, 1)

    def forward(self, states):
        x = F.relu(self.f1(states))
        x = F.relu(self.f2(x))
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, state, slot_mask, greedy=False):
        logits, value = self.forward(state)
        probs = torch.softmax(logits + EPS, dim=-1) * slot_mask
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = Categorical(probs)
        if greedy:
            a = torch.argmax(probs, dim=-1)
        else:
            a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, value, probs



# The remaining logic for state construction, slot assignment, frame rollout,
# .dat and .npy file generation, average AoI tracking, belief model for decoding probability,
# and plotting routines should now be added below step-by-step.

# Confirm once you're ready for the next complete segment: full rollout + .dat/.npy tracking and plots.
# Full working version of PD_NOMA_AoI.py with:
# - complete slot-level telemetry logging
# - extended state representation (energy, distance, time since last decode, belief)
# - saving .dat/.npy at slot-level and per-episode
# - per-user average AoI, system AoI tracking
# - plotting integrat


# PPOAgent, assign_slots_for_frame, plot_all_users_aoi(), and plot_all_users_energy()
# would also be included from your finalized PPO policy and visual functions.
# You can integrate these plots after training loop using:
#   plot_all_users_aoi(telemetry, num_slots)
#   plot_all_users_energy(telemetry, num_slots)

# Next step: plug this Telemetry + UserState in your main training loop.
# On every slot:
#   - update UserState belief
#   - log to telemetry
# At episode end:
#   - call telemetry.save_episode_npy()
#   - call telemetry.save_slotwise_dat()
#   - compute average AoI
#   - plot final timelines

# aoi_ppo_pdnoma_full.py
import csv
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.strategies.tree import greedy
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import os, json, numpy as np
from datetime import datetime
from telemetry_flush import new_buffer, append_slot, flush_episode, finalize_run_and_write_final

#from AOI_PPO_PDNOMA import num_frames, num_users


# ------------------- Repro -------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------- Environment params (yours) -------------------
num_slots          = 1
frames_per_episode = 1000
num_episodes       = 15
M_total            = 8
K_clusters         = 1
KF_clusters        = 4
K_r_user           = 12.0
alpha_c            = 0.25
beta_c             = 1.0
delta_wet          = 0.003
delta_wit          = 0.003
P_HAP              = 1.0
noise_pow          = 0.002
gamma_th_db        = 0
gamma_th           = 10 ** (gamma_th_db / 10.0)
v_L                = 0.5
v_H                = gamma_th * (1.0 + v_L)
drop_prob          = 0.6
D                  = 8.0
tau                = D / np.sqrt(2.0)  # near/far threshold by your spec
battery_max = 2

# ------------------- PPO params (plottable & stable) -------------------
state_dim    = 12        # [A_near, A_far, dA_near, dA_far, success_near, success_far]
lr           = 1e-2     # stable default
ppo_epochs   = 2
batch_size_I   = 200
gamma_I        = 0.99
lam_I          = 0.9
clip_range_I   = 0.1
ent_coef_I     = 0.01
vf_coef_I      = 0.5
max_grad_norm_I= 0.1
epsilon = 0.5

EPS = 1e-8

# ---- Early stop configuration ----
PI_TARGET      = 0.001     # policy loss target center
PI_TOL         = 0.005    # tolerance (±)
PI_PATIENCE    = 2        # consecutive episodes inside window
SAVE_NAME      = "best_policy.pt"
EVAL_EPISODES  = 30      # (used later if you run a frozen-policy eval)



#### _________________ Greedy Actions________ ####

# set once before training
epsilon_start = 0.25   # 20% random at the beginning
epsilon_final = 0.005   # floor
decay_steps   = 500000 # decisions/frames to reach the floor (tune as you like)
global_step = 0  # increment this once per decision (or per slot), not per episode

import os, datetime, numpy as np

def save_run_settings(
    out_dir,
    num_slots, frames_per_episode, num_episodes,
    M_total, K_clusters, KF_clusters, K_r_user,
    alpha_c, beta_c, delta_wet, delta_wit, P_HAP, noise_pow,
    gamma_th_db, v_L, drop_prob, D, tau,
    state_dim, lr, ppo_epochs, batch_size_I, gamma_I, lam_I,
    clip_range_I, ent_coef_I, vf_coef_I, max_grad_norm_I, epsilon,
    EPS, PI_TARGET, PI_TOL, PI_PATIENCE, SAVE_NAME, EVAL_EPISODES,
    epsilon_start, epsilon_final, decay_steps, global_step, max_battery=battery_max
):
    """
    Save current environment + PPO settings to a text file in the same directory.
    Creates:  <out_dir>/run_settings.txt
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "run_settings.txt")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gamma_th = 10 ** (gamma_th_db / 10.0)
    v_H = gamma_th * (1.0 + v_L)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"### Simulation Run Settings — {now}\n\n")

        f.write("# ------------------- Environment Params -------------------\n")
        f.write(f"max_battery       = {max_battery}\n")
        f.write(f"num_slots          = {num_slots}\n")
        f.write(f"frames_per_episode = {frames_per_episode}\n")
        f.write(f"num_episodes       = {num_episodes}\n")
        f.write(f"M_total            = {M_total}\n")
        f.write(f"K_clusters         = {K_clusters}\n")
        f.write(f"KF_clusters        = {KF_clusters}\n")
        f.write(f"K_r_user           = {K_r_user}\n")
        f.write(f"alpha_c            = {alpha_c}\n")
        f.write(f"beta_c             = {beta_c}\n")
        f.write(f"delta_wet          = {delta_wet}\n")
        f.write(f"delta_wit          = {delta_wit}\n")
        f.write(f"P_HAP              = {P_HAP}\n")
        f.write(f"noise_pow          = {noise_pow}\n")
        f.write(f"gamma_th_db        = {gamma_th_db}\n")
        f.write(f"gamma_th           = {gamma_th:.6f}\n")
        f.write(f"v_L                = {v_L}\n")
        f.write(f"v_H                = {v_H:.6f}\n")
        f.write(f"drop_prob          = {drop_prob}\n")
        f.write(f"D                  = {D}\n")
        f.write(f"tau                = {tau}\n\n")

        f.write("# ------------------- PPO Params -------------------\n")
        f.write(f"state_dim          = {state_dim}\n")
        f.write(f"lr                 = {lr}\n")
        f.write(f"ppo_epochs         = {ppo_epochs}\n")
        f.write(f"batch_size_I       = {batch_size_I}\n")
        f.write(f"gamma_I            = {gamma_I}\n")
        f.write(f"lam_I              = {lam_I}\n")
        f.write(f"clip_range_I       = {clip_range_I}\n")
        f.write(f"ent_coef_I         = {ent_coef_I}\n")
        f.write(f"vf_coef_I          = {vf_coef_I}\n")
        f.write(f"max_grad_norm_I    = {max_grad_norm_I}\n")
        f.write(f"epsilon            = {epsilon}\n")
        f.write(f"EPS                = {EPS}\n\n")

        f.write("# ---- Early Stop Configuration ----\n")
        f.write(f"PI_TARGET          = {PI_TARGET}\n")
        f.write(f"PI_TOL             = {PI_TOL}\n")
        f.write(f"PI_PATIENCE        = {PI_PATIENCE}\n")
        f.write(f"SAVE_NAME          = \"{SAVE_NAME}\"\n")
        f.write(f"EVAL_EPISODES      = {EVAL_EPISODES}\n\n")

        f.write("# ---- Greedy ε-Greedy Actions ----\n")
        f.write(f"epsilon_start      = {epsilon_start}\n")
        f.write(f"epsilon_final      = {epsilon_final}\n")
        f.write(f"decay_steps        = {decay_steps}\n")
        f.write(f"global_step        = {global_step}\n")

    print(f"[INFO] Saved run settings → {path}")
    return path

def epsilon_schedule(step):
    # linear decay
    if step >= decay_steps:
        return epsilon_final
    frac = 1.0 - (step / decay_steps)
    return epsilon_final + (epsilon_start - epsilon_final) * frac
# ------------------- Utility -------------------
def masked_softmax(logits, mask, dim=-1):
    """Numerically-stable masked softmax: mask 1=valid, 0=invalid."""
    mask = mask.to(logits.dtype)
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits + (mask + EPS).log()  # add log(mask) ~ 0 or -inf
    # Alternatively: masked = logits.masked_fill(mask < 0.5, neg_inf)
    return torch.softmax(masked, dim=dim)

def rician_power(K_r):
    """|h|^2 for Rician h = sqrt(K/(K+1)) + sqrt(1/(K+1))*g, g~CN(0,1)."""
    LOS = math.sqrt(K_r / (1.0 + K_r))
    NLOS = math.sqrt(1.0 / (1.0 + K_r))
    g = (np.random.normal(0.0, 1.0) + 1j*np.random.normal(0.0, 1.0)) #/ math.sqrt(2.0)
    h = LOS + NLOS * g
    return float((abs(h) ** 2))

def sepl_gain(d, alpha_c, beta_c):
    """SEPL large-scale power gain: exp(-alpha_c * d^beta_c)."""
    return math.exp(-alpha_c * (d ** beta_c))

# ------------------- User state -------------------
class UserState:
    def __init__(self, uid, d_init, energy_per_slot = 0.01, max_bat=0.004):
        self.uid = uid
        self.d = float(d_init)     # last-known distance (updated on success)
        self.h = None              # complex small-scale (sampled per frame)
        self.gamma = None          # combined gain = sepl_gain(d)*|h|^2
        self.battery = 0.001
        self.aoi = 0
        #self.aoi_prev = 0
        self.aoi_sum = 0.0
        self.aoi_count = 0
        self.decode = 0.0
        self.v_assigned_target = None  # the v (v_H or v_L) used THIS frame if scheduled
        self.decode_prev = 0  # to store previous decoded state
        self.max_bat = float(max_bat)
        self.harvested = 0
        # --- NEW attributes for advanced policies ---
        self.last_slot_used = -1  # which slot this user was last assigned
        self.last_decoded = 0  # last decode success indicator (0/1)
        self.succ_ema = 0.0  # running exponential success average
        self.prev_frame_aoi_start = 1
        self.prev_frame_aoi_end = 1
        self.delta_aoi_prev = 0.0  # (end - start) of the *previous* frame
        self.last_decoded_slot = 0

        # Belief system
        self.belief_energy = 0.0
        self.energy_per_slot = energy_per_slot

        self.scheduled_count = 0
        self.success_count = 0
        self.belief_psuccess = 0.5  # initial belief

        self.scheduled_count = 0
        self.success_count = 0
        self.belief_psuccess = 0.5  # initial belief

    def update_belief_energy(self, current_slot):
        slots_since_last = current_slot - self.last_decoded_slot
        self.belief_energy += slots_since_last * self.energy_per_slot
        self.last_decoded_slot = current_slot  # reset marker
        return self.belief_energy

    def update_belief_psuccess(self, decoded):
        self.scheduled_count += 1
        if decoded:
            self.success_count += 1
        if self.scheduled_count > 0:
            self.belief_psuccess = self.success_count / self.scheduled_count
        else:
            self.belief_psuccess = 0.0

    def __repr__(self):
        return f"{self.uid}"

    def sample_channel_and_gamma(self, K_r, alpha_c, beta_c):
        # Rician small-scale
        LOS = math.sqrt(K_r / (1.0 + K_r))
        NLOS = math.sqrt(1.0 / (1.0 + K_r))
        g = (np.random.normal(0.0, 1.0) + 1j*np.random.normal(0.0, 1.0)) #/ math.sqrt(2.0)
        self.h = LOS + NLOS * g
        self.gamma = sepl_gain(self.d, alpha_c, beta_c) * (abs(self.h) ** 2)
        return self.gamma

    def compute_energy_harvested(self, gamma_val, delta_wet, P_HAP, alpha0=0.826, alpha1=0.399):
        # Nonlinear EH model
        num = alpha0 * P_HAP * delta_wet * gamma_val
        den = alpha1 * P_HAP * gamma_val + alpha1
        return num / max(den, EPS)

    def tx_power_for_target(self, v_target):
        # P_tx = v / gamma
        if self.gamma is None or self.gamma <= 0.0:
            return float('inf')
        return float(v_target) / float(self.gamma)

    def battery_check(self):
        self.battery = float(np.clip(self.battery, 0.0, self.max_bat))

    def spend_tx_energy(self, delta_wit, P_tx):
        #P_tx = self.tx_power_for_target(v_target)
        need = P_tx * delta_wit
        if not np.isfinite(need):
            return False
        if self.battery + 1e-12 >= need:
            self.battery -= need
            return True
        return False

    def tally_aoi(self):
        self.aoi_sum += self.aoi
        self.aoi_count += 1

    def avg_aoi(self):
        return self.aoi_sum / max(self.aoi_count, 1)

    def update_belief(self, slot_idx, energy_per_slot=0.01):
        slots_since_last = slot_idx - self.last_decoded_slot
        self.belief_energy = self.battery + slots_since_last * energy_per_slot

    def compute_success_prob(self):
        return 1.0 / (1.0 + np.exp(5 * self.d / max(1e-5, self.belief_energy)))


# ------------------- Inter-cell interference (cross-link only) -------------------
def intercell_interference(K_clusters, alpha_c, beta_c, v_H, v_L, d_cross_rng=(40.0, 60.0), K_r_cross=KF_clusters):
    """
    Sum of power from (K_clusters-1) other clusters, each with one high & one low user.
    We compute received power at our HAP using SEPL(d_cross)*|h|^2 multiplied by that user's
    target received power v_H / v_L (constant-power interferers at our HAP viewpoint).
    """
    I_total = 0.0
    for _ in range(K_clusters - 1):
        # high-power interferer
        dH = float(np.random.uniform(*d_cross_rng))
        gH = sepl_gain(dH, alpha_c, beta_c) * rician_power(K_r_cross)
        I_total += v_H * gH
        # low-power interferer
        dL = float(np.random.uniform(*d_cross_rng))
        gL = sepl_gain(dL, alpha_c, beta_c) * rician_power(K_r_cross)
        I_total += v_L * gL
    return I_total

# ------------------- PD-NOMA decoding -------------------
def decode_pair_pd_noma(uH, uL, v_H, v_L, gamma_th, noise_pow, K_clusters,
                        alpha_c, beta_c, delta_wet, delta_wit, P_HAP):
    """
    1) Both harvest using their gamma
    2) Attempt decode high user first: SINR_H = v_H / (v_L + I + N0)
    3) If success, low user SINR_L = v_L / (I + N0)
    4) Energy feasibility: must pay UL energy P_tx * delta_wit
    """
    # WET
    #E_H = uH.compute_energy_harvested(uH.gamma, delta_wet, P_HAP)
    #E_L = uL.compute_energy_harvested(uL.gamma, delta_wet, P_HAP)
    #uH.battery += E_H
    #uH.battery_check()
    #uL.battery += E_L
    #uL.battery_check()

    # Inter-cell interference (common for both within slot)
    I_cross = intercell_interference(K_clusters, alpha_c, beta_c, v_H, v_L)
    feas_L = False
    feas_H = False
    req_EL = False
    req_EH = False
    # High user first
    Ptx_H = uH.tx_power_for_target(v_H)
    SINR_H = v_H / max(v_L + I_cross + noise_pow, EPS)  # PD-NOMA model at HAP
    feas_H = (uH.battery >= Ptx_H * delta_wit) and np.isfinite(Ptx_H) and (SINR_H >= gamma_th)

    dec_H = False
    if feas_H:
        # spend energy and declare success
        if uH.spend_tx_energy(delta_wit, Ptx_H):
            dec_H = True
            req_EH = True

    # Low user (conditioned on success of high)
    dec_L = False
    SINR_L = 0.0
    Ptx_L = uL.tx_power_for_target(v_L)
    SINR_L = v_L / max(I_cross + noise_pow, EPS)
    feas_L = (uL.battery >= Ptx_L * delta_wit) and np.isfinite(Ptx_L) and (SINR_L >= gamma_th)


    if dec_H:
        if feas_L:
            if uL.spend_tx_energy(delta_wit, Ptx_L):
                dec_L = True
                req_EL = True

    #return dec_H, dec_L, SINR_H, SINR_L
    return dec_H, dec_L, SINR_H, SINR_L, uH.battery, uL.battery, req_EH, req_EL

# ------------------- OMA fallback (single user in slot) -------------------
def decode_single_oma(u, v_target, gamma_th, noise_pow, K_clusters,
                      alpha_c, beta_c, delta_wet, delta_wit, P_HAP):
    # WET
    #E = u.compute_energy_harvested(u.gamma, delta_wet, P_HAP)
    #u.battery += E
    #u.battery_check()
    # Inter-cell interference still exists (two users/cluster in others)
    I_cross = intercell_interference(K_clusters, alpha_c, beta_c, v_H, v_L)
    Ptx = u.tx_power_for_target(v_target)
    SINR = v_target / max(I_cross + noise_pow, EPS)
    feas = (u.battery >= Ptx * delta_wit) and np.isfinite(Ptx) and SINR >= gamma_th
    dec = False
    req_EH = False
    if feas:
        if u.spend_tx_energy(delta_wit, Ptx):
            dec = True
            req_EH = True
    #return dec, SINR
    return dec, SINR, u.battery, req_EH


def split_groups_by_distance(users, tau):
    near = [u for u in users if u.d <= tau]
    far  = [u for u in users if u.d >  tau]
    return near, far

def init_allowed_idle(near, far, num_slots):
    rng = np.random.default_rng()
    near = near.copy()
    far  = far.copy()
    rng.shuffle(near)
    rng.shuffle(far)
    allowed_near = near[:num_slots]
    idle_near    = near[num_slots:]
    allowed_far  = far[:num_slots]
    idle_far     = far[num_slots:]
    return allowed_near, idle_near, allowed_far, idle_far

def reconcile_after_successes(all_users, allowed_near, idle_near, allowed_far, idle_far, tau):
    """
    Users that succeeded may change distance for next frame (e.g., new measurement).
    Here we choose to re-sample distance for successes to reflect mobility; feel free to change.
    Then re-split all users into near/far, rebuild allowed/idle with best effort to keep sizes.
    """
    near, far = split_groups_by_distance(all_users, tau)

    # Preserve allowed counts if possible
    def rebuild(allowed_old, idle_old, new_pool):
        # Keep those still in pool; fill up with others from pool
        keep = [u for u in allowed_old if u in new_pool]
        pool_rest = [u for u in new_pool if u not in keep]
        rng = np.random.default_rng()
        rng.shuffle(pool_rest)
        allowed_new = keep.copy()
        while len(allowed_new) < num_slots and pool_rest:
            allowed_new.append(pool_rest.pop())
        idle_new = pool_rest + [u for u in idle_old if u in new_pool and u not in allowed_new]
        return allowed_new, idle_new

    allowed_near, idle_near = rebuild(allowed_near, idle_near, near)
    allowed_far,  idle_far  = rebuild(allowed_far,  idle_far,  far)

    return allowed_near, idle_near, allowed_far, idle_far

def update_group_after_frame_old(allowed, idle, successes, drop_prob, rng):
    """
    Drop a fraction of successful users and replace with idle users.
    """
    if len(successes) == 0 or len(idle) == 0:
        return allowed, idle
    n_drop = math.ceil(drop_prob * len(successes))#int(drop_prob * len(successes))
    if n_drop <= 0:
        return allowed, idle
    drop = list(rng.choice(successes, size=min(n_drop, len(successes)), replace=False))
    # filter in case some duplicates
    drop = [u for u in drop if u in allowed]
    if len(drop) == 0:
        return allowed, idle
    add = list(rng.choice(idle, size=min(len(drop), len(idle)), replace=False))
    new_allowed = [u for u in allowed if u not in drop] + add
    new_idle    = [u for u in idle if u not in add] + drop
    return new_allowed, new_idle


def update_group_after_frame(allowed, idle, successes, drop_prob, rng, users_by_uid=None):
    """
    Drop a fraction of successful users and replace with the highest-AoI idle users.

    Parameters
    ----------
    allowed : list[UserState] | list[int]
    idle    : list[UserState] | list[int]
    successes : list[UserState] | list[int]
    drop_prob : float in [0,1]
    rng : np.random.Generator
    users_by_uid : dict[int, UserState] | None
        Only needed if your lists contain integer uids instead of UserState objects.

    Returns
    -------
    (new_allowed, new_idle)
    """

    # --- helper to get the UserState object regardless of representation ---
    def U(x):
        if users_by_uid is not None and not hasattr(x, "aoi"):
            # x is an int uid → look up object
            return users_by_uid[x]
        return x  # already a UserState

    if not successes or not idle:
        return allowed, idle

    # How many to drop (Alg.1 style)
    n_drop = math.ceil(drop_prob * len(successes))
    if n_drop <= 0:
        return allowed, idle

    # Consider only successful users that are still in allowed
    pool = [s for s in successes if s in allowed]
    if not pool:
        return allowed, idle

    n_drop = min(n_drop, len(pool))
    drop = list(rng.choice(pool, size=n_drop, replace=False))

    # ----- choose additions: highest-AoI first from idle -----
    # Fair tiebreak: randomize first, then stable sort by AoI desc
    if len(idle) > 1:
        perm = rng.permutation(len(idle))
        idle_shuffled = [idle[i] for i in perm]
    else:
        idle_shuffled = list(idle)

    # Fetch AoI directly from UserState
    def aoi_of(x):
        return float(U(x).aoi)  # UserState.aoi

    idle_sorted = sorted(idle_shuffled, key=aoi_of, reverse=True)
    add = idle_sorted[:min(len(drop), len(idle_sorted))]

    # ----- rebuild groups (preserve order of survivors in allowed) -----
    drop_set = set(drop)
    add_set  = set(add)

    new_allowed = [u for u in allowed if u not in drop_set] + list(add_set)
    new_idle    = [u for u in idle   if u not in add_set]  + list(drop_set)

    return new_allowed, new_idle


# ------------------- Pairing -------------------
def make_pairs(allowed_near, allowed_far, n_slots):
    """
    Pair min(len(near), len(far), n_slots) pairs.
    If shortage on one side, leftover singles will be OMA in those slots.
    """
    rng = np.random.default_rng()
    near = allowed_near.copy()
    far  = allowed_far.copy()
    rng.shuffle(near)
    rng.shuffle(far)

    n_pairs = min(len(near), len(far), n_slots)
    pairs = []
    for i in range(n_pairs):
        pairs.append((near[i], far[i]))
    # singles left?
    singles = []
    if n_pairs < n_slots:
        # Fill remaining slots with singles from whichever side has extras
        extra_near = near[n_pairs:]
        extra_far  = far[n_pairs:]
        pool = extra_near + extra_far
        while len(pairs) + len(singles) < n_slots and pool:
            singles.append(pool.pop())
    return pairs, singles

# ------------------- PPO model & buffer -------------------


class RolloutBuffer:
    def __init__(self, capacity, state_dim, n_slots, device):
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.masks  = torch.zeros((capacity, n_slots),    dtype=torch.float32, device=device)
        self.actions= torch.zeros((capacity,),            dtype=torch.long,    device=device)
        self.logps  = torch.zeros((capacity,),            dtype=torch.float32, device=device)
        self.values = torch.zeros((capacity,),            dtype=torch.float32, device=device)
        self.rewards= torch.zeros((capacity,),            dtype=torch.float32, device=device)
        self.dones  = torch.zeros((capacity,),            dtype=torch.float32, device=device)
        self.ptr = 0

    def add(self, s, m, a, logp, v, r, done):
        if self.ptr >= self.capacity:
            return False
        self.states[self.ptr]  = s.detach()
        self.masks[self.ptr]   = m.detach()
        self.actions[self.ptr] = a.detach()
        self.logps[self.ptr]   = logp.detach()
        self.values[self.ptr]  = v.detach()
        self.rewards[self.ptr] = float(r)
        self.dones[self.ptr]   = float(done)
        self.ptr += 1
        return True

    def __len__(self):
        return self.ptr

    def clear(self):
        self.ptr = 0

def compute_gae(rewards, values, dones, gamma=gamma_I, lam=lam_I):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_v = 0.0 if t == T - 1 else values[t+1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
        #adv[t] = (adv[t] - adv.mean()) / (adv.std() + 1e-8)
    returns = adv + values
    return adv, returns

def ppo_update(policy, buffer, optimizer,
               epochs = ppo_epochs, batch_size=batch_size_I,
               clip_range=clip_range_I, ent_coef=ent_coef_I, vf_coef=ent_coef_I, max_grad_norm=max_grad_norm_I,
               gamma=gamma_I, lam=lam_I):
    policy.train()
    T = len(buffer)
    if T == 0:
        return dict(pi_loss=0.0, v_loss=0.0, ent=0.0, kl=0.0)

    advantages, returns = compute_gae(buffer.rewards[:T], buffer.values[:T], buffer.dones[:T],
                                      gamma=gamma, lam=lam)
    # normalize advantages
    adv_std = advantages.std().clamp_min(1e-6)
    advantages = (advantages - advantages.mean()) / adv_std

    idxs = torch.arange(T, device=device)
    pi_losses = []
    v_losses = []
    ents = []
    kls = []

    for _ in range(epochs):
        perm = idxs[torch.randperm(T)]
        for start in range(0, T, batch_size):
            mb = perm[start:start+batch_size]
            s   = buffer.states[mb]
            msk = buffer.masks[mb]
            a   = buffer.actions[mb]
            old_logp = buffer.logps[mb]
            ret = returns[mb]
            adv = advantages[mb]

            logits, v = policy(s)
            probs = masked_softmax(logits, msk, dim=-1).clamp_min(EPS)
            dist = Categorical(probs)
            logp = dist.log_prob(a)
            entropy = dist.entropy().mean()

            # ratio
            ratio = (logp - old_logp).exp().clamp(0.0, 10.0)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * adv
            pi_loss = -torch.min(surr1, surr2).mean()

            v_loss = F.mse_loss(v, ret)

            loss = pi_loss + vf_coef * v_loss - ent_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                # sample KL approximation (old vs new) on mb
                # KL(P_old || P_new) ~ E[logp_old - logp_new]
                kl = (old_logp - logp).mean().clamp_min(0.0)

            pi_losses.append(pi_loss.item())
            v_losses.append(v_loss.item())
            ents.append(entropy.item())
            kls.append(kl.item())

    return dict(
        pi_loss=float(np.mean(pi_losses) if pi_losses else 0.0),
        v_loss=float(np.mean(v_losses) if v_losses else 0.0),
        ent=float(np.mean(ents) if ents else 0.0),
        kl=float(np.mean(kls) if kls else 0.0),
    )

@torch.no_grad()
def assign_slots_ppo(policy, device, users, num_slots, build_pair_state, candidate_pairs):
    """
    Returns slot_map: {slot_idx: ("pair",(uN,uF), slot_idx)}
    Uses current policy.act(...) with a mask so no slot is double-used.
    """
    slot_map = {}
    # start all slots as free
    slot_mask = torch.ones(1, num_slots, device=device)

    for (uN, uF) in candidate_pairs:
        # build state exactly like training
        s_np = build_pair_state(uN, uF)[None, :]
        s = torch.from_numpy(s_np).to(device)
        # act (sample or argmax per your flag; here we sample)
        a, logp, v, _ = policy.act(s, slot_mask, greedy=False)
        slot_idx = int(a.item())

        if slot_idx not in slot_map:
            slot_map[slot_idx] = ("pair", (uN, uF), slot_idx)
            slot_mask[0, slot_idx] = 0.0  # mark taken
        # if the chosen slot already taken, you can optionally
        # assign to the next free slot to avoid dropping the pair:
        # else:
        #     free = _free_slots(num_slots, slot_map)
        #     if free:
        #         s2 = int(free[0])
        #         slot_map[s2] = ("pair",(uN,uF), s2)
        #         slot_mask[0, s2] = 0.0

        if len(slot_map) >= num_slots:
            break
    return slot_map


def build_pair_state(uN, uF):
    import numpy as np

    def _safe(x, lo=-1e6, hi=1e6):
        x = float(x)
        if not np.isfinite(x): return 0.0
        return float(np.clip(x, lo, hi))

    A_n, A_f = _safe(uN.aoi, 0, 1e6), _safe(uF.aoi, 0, 1e6)
    dA_n, dA_f = _safe(uN.delta_aoi_prev, -1e6, 1e6), _safe(uF.delta_aoi_prev, -1e6, 1e6)
    dec_n, dec_f = _safe(uN.last_decoded_slot, 0, 1e6), _safe(uF.last_decoded_slot, 0, 1e6)
    dist_n, dist_f = _safe(uN.d, 0, 1e6), _safe(uF.d, 0, 1e6)
    belief_e_n, belief_e_f = _safe(uN.belief_energy, 0, 1e6), _safe(uF.belief_energy, 0, 1e6)
    psucc_n, psucc_f = _safe(uN.belief_psuccess, 0, 1), _safe(uF.belief_psuccess, 0, 1)

    s = np.array([
        A_n, A_f, dA_n, dA_f, dec_n, dec_f, dist_n, dist_f,
        belief_e_n, belief_e_f, psucc_n, psucc_f
    ], dtype=np.float32)

    # last line of defense:
    s = np.nan_to_num(s, nan=0.0, posinf=1e6, neginf=-1e6)
    return s


def make_run_dir(M_total, num_slots):
    #stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"AoI_U{M_total}_S{num_slots}_SLT"
    out = os.path.join(name)
    os.makedirs(out, exist_ok=True)
    return out

def to_slot_id(s, num_slots):
    import numpy as np
    import torch

    # PyTorch tensor cases
    if torch.is_tensor(s):
        # scalar tensor
        if s.dim() == 0:
            return int(s.item())
        # shape (1,)
        if s.dim() == 1 and s.numel() == 1:
            return int(s.item())
        # shape (num_slots,) -> take argmax (works for logits, probs, or one-hot)
        if s.dim() == 1 and s.numel() == num_slots:
            return int(torch.argmax(s).item())
        # boolean/one-hot with a single nonzero entry
        nz = torch.nonzero(s, as_tuple=False).view(-1)
        if nz.numel() == 1:
            return int(nz.item())
        raise ValueError(f"Cannot convert tensor of shape {tuple(s.shape)} to slot id")

    # list/tuple/numpy
    if isinstance(s, (list, tuple, np.ndarray)):
        arr = np.array(s)
        if arr.size == 1:
            return int(arr.item())
        if arr.ndim == 1 and arr.size == num_slots:
            return int(int(arr.argmax()))
        nz = np.flatnonzero(arr)
        if nz.size == 1:
            return int(nz[0])
        raise ValueError(f"Cannot convert array of shape {arr.shape} to slot id")

    # fallback (int-like)
    return int(s)

import os
import re
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

EPS = 1e-8

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, n_slots):
        super().__init__()
        self.n_slots = n_slots
        self.f1 = nn.Linear(state_dim, 128)
        self.f2 = nn.Linear(128, 128)
        self.pi = nn.Linear(128, n_slots)
        self.v  = nn.Linear(128, 1)

    def forward(self, states):
        x = F.relu(self.f1(states))
        x = F.relu(self.f2(x))
        logits = self.pi(x)
        value  = self.v(x).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, state, slot_mask, greedy=False):
        """
        Safe masked policy:
          - sanitize state/logits (nan_to_num)
          - masked softmax (mask=0 → -inf) so no renorm divide-by-zero
          - fallback to uniform over available slots if all masked or bad
        """
        # 1) sanitize state (in case env/state construction produced NaN/Inf)
        state = torch.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)

        logits, value = self.forward(state)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) masked softmax (stable): invalid slots get -inf before softmax
        mask = slot_mask.to(dtype=logits.dtype)
        # if mask has no valid entries, we’ll fix below
        very_neg = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(mask < 0.5, very_neg)

        # subtract max for stability
        max_logits = masked_logits.max(dim=-1, keepdim=True).values
        masked_logits = masked_logits - max_logits

        probs = torch.softmax(masked_logits, dim=-1)  # already respects mask
        probs = torch.nan_to_num(probs, nan=0.0)

        # 3) fallback if everything was masked or probs degenerated
        sum_probs = probs.sum(dim=-1, keepdim=True)
        bad = (sum_probs <= 0) | ~torch.isfinite(sum_probs)
        if bad.any():
            # try uniform over available slots
            avail = (mask > 0.5).to(probs.dtype)
            denom = avail.sum(dim=-1, keepdim=True).clamp_min(1.0)
            probs = avail / denom
            # if STILL all zeros (no available slots), fall back to argmax(logits) ignoring mask
            still_bad = (probs.sum(dim=-1, keepdim=True) <= 0)
            if still_bad.any():
                # one-hot at global argmax (last resort)
                idx = torch.argmax(logits, dim=-1)
                probs = torch.zeros_like(logits)
                probs.scatter_(1, idx.unsqueeze(-1), 1.0)

        dist = Categorical(probs)
        a = torch.argmax(probs, dim=-1) if greedy else dist.sample()
        logp = dist.log_prob(a)
        return a, logp, value, probs


# telemetry_unified.py
import os, re, gc, glob, csv, pickle, tempfile
import numpy as np
from collections import defaultdict

class Telemetry:
    """
    Unified telemetry logger with episode-wise flushing.
    Data shape (in RAM and on disk):
        by_uid: dict[int uid] -> list[ row_dict ]
    Row keys (exactly as requested):
        'ep','frame','slot','uid','uid_str','step','kind','pd_role',
        'scheduled','decoded','required','aoi','battery','harvested','sinr','distance'
    """

    def __init__(self):
        self.by_uid = defaultdict(list)
        self._tick = 0
        self._frame_scratch = {}

    # ------------- ID parsing & step counter -------------
    @staticmethod
    def _parse_uid(u):
        """
        Accepts a UserState (with .uid like 'U32'), a string 'U32'/'32',
        or an int 32. Returns (uid_num:int, uid_str:str).
        """
        # UserState
        raw = getattr(u, "uid", u)

        # string like 'U2' or '2'
        if isinstance(raw, str):
            m = re.search(r"\d+", raw)
            if m:
                n = int(m.group(0))
                return n, f"U{n}"
            raise ValueError(f"String uid must contain a number: {raw!r}")

        # integer
        if isinstance(raw, (int, np.integer)):
            n = int(raw)
            return n, f"U{n}"

        raise ValueError(f"Invalid uid: {u!r} (expected UserState/int/'U#')")

    def tick(self):
        self._tick += 1
        return self._tick

    # ------------- Logging (keep your current call signature) -------------
    def log_user(self, ep, u, frame, slot, kind, sinr, battery,
                 harvested, required, decoded, aoi, distance,
                 scheduled, pd_role):
        """
        Called from your slot loop, e.g.:
          telemetry.log_user(ep=ep, u=uN, frame=frame, slot=sl, kind="PDNOMA",
                             sinr=sinr_H, battery=uN.battery, harvested=uN.harvested,
                             required=reqH, decoded=uN.decode, aoi=uN.aoi, distance=uN.d,
                             scheduled=1, pd_role="NOMA-H")
        """
        uid_num, uid_str = self._parse_uid(u)
        row = {
            "ep":        int(ep),
            "frame":     int(frame),
            "slot":      int(slot),
            "uid":       uid_num,
            "uid_str":   uid_str,
            "step":      self.tick(),
            "kind":      (str(kind) if kind else ""),
            "pd_role":   (str(pd_role) if pd_role else ""),
            "scheduled": int(bool(scheduled)),
            "decoded":   int(bool(decoded)),
            "required":  int(bool(required)),
            "aoi":       float(aoi) if aoi is not None else 0.0,
            "battery":   float(battery) if battery is not None else 0.0,
            "harvested": float(harvested) if harvested is not None else 0.0,
            "sinr":      float(sinr) if sinr is not None else 0.0,
            "distance":  float(distance) if distance is not None else 0.0,
        }
        self.by_uid[uid_num].append(row)

    # ------------- Optional per-frame helpers (kept for compatibility) -------------
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

    def clear_frame(self, ep=None, frame=None):
        """
        Compatibility: if ep & frame given, remove rows belonging to that frame.
        If called with no args (legacy), just clear scratch (not by_uid).
        """
        if ep is None or frame is None:
            # legacy behavior: do not delete logged rows
            self._frame_scratch.clear()
            return 0

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

    def clear_all(self):
        self.by_uid.clear()
        self._frame_scratch.clear()
        gc.collect()

    # ------------- Episode flushing to chunks -------------
    def flush_episode(self, run_dir, M_total, num_slots, ep):
        """
        Save one episode chunk to:
          {run_dir}/chunks/slotwise_data_ep{ep:04d}_U{M_total}S{num_slots}.npy
        Then clear in-RAM buffer.
        """
        if not self.by_uid:
            return
        chunks_dir = os.path.join(run_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        out_path = os.path.join(
            chunks_dir, f"slotwise_data_ep{int(ep):04d}_U{M_total}S{num_slots}.npy"
        )
        self._atomic_save_dict(out_path, dict(self.by_uid))
        self.by_uid.clear()
        gc.collect()
        print(f"[FLUSH] episode={ep} → {out_path}")

    # ------------- Final assembly (write the SAME final file your plots expect) -------------
    def finalize_run(self, run_dir, M_total, num_slots,
                     final_filename=None, keep_chunks=True):
        """
        Merge all chunks (+ leftovers in RAM) into:
            slotwise_dataU{M_total}S{num_slots}.npy
        Structure: dict[uid:int] -> list[row_dict]
        """
        if final_filename is None:
            final_filename = f"slotwise_dataU{M_total}S{num_slots}.npy"
        final_path = os.path.join(run_dir, final_filename)

        merged = defaultdict(list)
        chunks_dir = os.path.join(run_dir, "chunks")
        pattern = os.path.join(chunks_dir, f"slotwise_data_ep*_U{M_total}S{num_slots}.npy")
        chunk_paths = sorted(glob.glob(pattern))

        for cp in chunk_paths:
            try:
                d = np.load(cp, allow_pickle=True).item()
                for k, v in d.items():
                    merged[int(k)].extend(v)
            except Exception as e:
                print(f"[WARN] failed to read chunk {cp}: {e}")

        if self.by_uid:
            for k, v in self.by_uid.items():
                merged[int(k)].extend(v)
            self.by_uid.clear()

        self._atomic_save_dict(final_path, dict(merged))
        print(f"[FINAL] wrote {final_path}  (uids={len(merged)})")

        if not keep_chunks and chunk_paths:
            for cp in chunk_paths:
                try: os.remove(cp)
                except Exception: pass
            try: os.rmdir(chunks_dir)
            except Exception: pass

    # ------------- Legacy save methods (both write same structure) -------------
    def save_episode_npy(self, run_dir, filename="episode_data.npy"):
        """
        Legacy name. Writes the current in-RAM dict (by_uid) as a single .npy file.
        For compatibility, same structure as save_slotwise_npy().
        """
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        self._atomic_save_dict(path, dict(self.by_uid))
        print(f"[SAVE] {path} (unified structure)")

    def save_slotwise_dat(self, run_dir, filename="slotwise_data.dat"):
        """
        Raw pickle (if you need it for debugging). Not used by the plots.
        """
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(self.by_uid, f)
        print(f"[SAVE] {path} (pickle)")

    def save_slotwise_npy(self, run_dir, filename="slotwise_data.npy"):
        """
        Legacy name. Writes the current in-RAM dict (by_uid) as a single .npy file.
        For compatibility, same structure as save_episode_npy().
        """
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, filename)
        self._atomic_save_dict(path, dict(self.by_uid))
        print(f"[SAVE] {path} (unified structure)")

    # ------------- Simple AOI utilities -------------
    def compute_average_aoi(self, total_slots):
        uid_avg = {}
        for uid, rows in self.by_uid.items():
            # robust against missing keys
            aoi_sum = sum(float(r.get("aoi", 0.0)) for r in rows)
            uid_avg[uid] = aoi_sum / max(int(total_slots), 1)
        return uid_avg

    def compute_system_aoi(self, total_slots):
        user_avgs = self.compute_average_aoi(total_slots)
        vals = [v for v in user_avgs.values() if np.isfinite(v)]
        return (sum(vals) / len(vals)) if vals else np.nan

    # ------------- helpers -------------
    @staticmethod
    def _atomic_save_dict(path, payload_dict):
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

#telemetry = Telemetry()
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


def assign_slots_for_frame(
    frame, ep, pairs, singles, num_slots, device, build_pair_state,
    policy=None, epsilon_schedule=None, global_step=None, greedy_act=False,
    tau=10.0, alpha=1.0, beta=0.1, greedy_mode="max", rng=None
):
    import numpy as np, torch
    rng = rng or np.random.default_rng()

    slot_mask = torch.ones((1, num_slots), dtype=torch.float32, device=device)
    assigned_slots = []
    states_to_buffer, masks_to_buffer = [], []
    values_to_buffer, logps_to_buffer, actions_to_buffer = [], [], []
    slot_map, idx_by_slot = {}, {}
    taken = set()

    def _push_buffers(s_tensor, cur_mask, v, logp, a_idx):
        k = len(states_to_buffer)
        states_to_buffer.append(s_tensor)
        masks_to_buffer.append(cur_mask.clone())
        values_to_buffer.append(v)
        logps_to_buffer.append(logp)
        actions_to_buffer.append(int(a_idx))
        return k

    # Build items list
    items = [("pair", p) for p in pairs] + [("single", s) for s in singles]
    if len(items) < num_slots:
        raise RuntimeError(
            f"assign_slots_for_frame: not enough items to fill {num_slots} slots "
            f"(got {len(items)}). Ensure pairs+singles cover capacity."
        )

    for typ, obj in items:
        if len(taken) >= num_slots:
            break

        # ---- unify per-iteration variables ----
        if typ == "pair":
            users = (obj[0], obj[1])          # tuple of 2 UserState
            is_pair = True
            s_np = build_pair_state(users[0], users[1])[None, :]
        else:
            users = (obj,)                    # tuple of 1 UserState
            is_pair = False
            s_np = build_pair_state(users[0], users[0])[None, :]

        s = torch.from_numpy(s_np).to(device)

        # ---- epsilon-greedy over PPO policy ----
        if epsilon_schedule is not None and global_step is not None:
            eps = float(epsilon_schedule(global_step))
            take_random = (rng.random() < eps)
            greedy_now = greedy_act or (not take_random)   # exploit if not exploring
        else:
            eps = 0.0
            greedy_now = greedy_act

        # ensure there's at least one free slot
        if torch.count_nonzero(slot_mask) == 0:
            # all taken (shouldn’t normally happen before the loop break)
            free = [ss for ss in range(num_slots) if ss not in taken]
            if not free:
                break
            # open one slot to avoid all-zero mask
            slot_mask[0, free[0]] = 1.0

        a, logp_t, v_t, _ = policy.act(s, slot_mask, greedy=greedy_now)
        slot_idx = int(a.item())

        # Log SAR decision (reward will be filled later)
        if 'sar_logger' in globals():
            sar_logger.log(ep, frame, slot_idx, s_np.squeeze().tolist(), slot_idx)


        # fallback if chosen slot already taken
        if slot_idx in taken:
            free = [ss for ss in range(num_slots) if ss not in taken]
            if not free:
                break
            slot_idx = int(min(free))

        assert slot_idx not in taken, f"[BUG] slot {slot_idx} already taken"

        k = _push_buffers(s, slot_mask, v_t, logp_t, slot_idx)

        try:
            slot_id = to_slot_id(slot_idx, num_slots)  # if you use it elsewhere
        except Exception:
            slot_id = slot_idx
        idx_by_slot[slot_id] = k

        # ---- record consistently using 'users' tuple ----
        if is_pair:
            assigned_slots.append(("pair", users, slot_idx))
            slot_map[slot_idx] = users              # (uN, uF)
            users[0].v_assigned_target = None
            users[1].v_assigned_target = None
        else:
            assigned_slots.append(("single", users[0], slot_idx))
            slot_map[slot_idx] = users[0]           # u
            users[0].v_assigned_target = None

        taken.add(slot_idx)
        slot_mask[0, slot_idx] = 0.0

        if global_step is not None:
            global_step += 1  # advance per decision (optional)

    # Final invariants
    if len(taken) != num_slots or len(slot_map) != num_slots:
        missing = [s for s in range(num_slots) if s not in slot_map]
        raise RuntimeError(
            f"assign_slots_for_frame: expected {num_slots} filled slots, "
            f"got {len(slot_map)}. Missing slots: {missing}"
        )
    assert set(slot_map.keys()) == set(range(num_slots)), "slot_map must have all slot keys"

    return (assigned_slots, slot_map,
            states_to_buffer, masks_to_buffer, values_to_buffer, logps_to_buffer, actions_to_buffer,
            idx_by_slot, slot_mask)

def compute_average_aoi(telemetry, num_slots, num_frames, out_dir):
    """
    Computes moving average AoI for each user and system at the end of an episode.
    Saves the result as .dat and .npy.
    """
    import numpy as np
    import os

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    avg_per_user = {}
    total_time = num_slots * num_frames

    for uid, rows in telemetry.by_uid.items():
        rows_sorted = sorted(rows, key=lambda r: r["step"])
        aoi_values = [r["aoi"] for r in rows_sorted]
        cumulative_sum = np.cumsum(aoi_values)
        time_indices = np.arange(1, len(aoi_values) + 1)
        moving_avg = cumulative_sum / time_indices
        final_avg = moving_avg[-1] if moving_avg.size > 0 else 0.0
        avg_per_user[f"U{uid}"] = final_avg

    system_avg = np.mean(list(avg_per_user.values()))

    # Save to .dat
    dat_path = os.path.join(out_dir, "avg_aoi_summary.dat")
    with open(dat_path, "w") as f:
        for uid, avg in avg_per_user.items():
            f.write(f"{uid}: {avg:.4f}\n")
        f.write(f"\nSystem Average AoI: {system_avg:.4f}\n")

    # Save to .npy
    npy_path = os.path.join(out_dir, "avg_aoi_per_user.npy")
    #np.save(npy_path, avg_per_user)

    print(f"[SAVE] Saved AoI summary to {dat_path}")
    return system_avg

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

def plot_all_users_avg_aoi_combined(data, num_slots, frames_per_episode, num_episodes,
                                     out_pdf="All_Users_MovingAvgAoI.pdf", out_dir="telemetry_plots"):
    import os, numpy as np, matplotlib as mpl, matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    # IEEE-style consistent formatting
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

    plt.figure(figsize=(6.6, 3.2))  # IEEE 2-column width

    uids = sorted(data.keys())
    max_time = 0
    for uid in uids:
        rows = sorted(data[uid], key=lambda r: (r["ep"], r["frame"], r["slot"]))
        t = []
        avg_aoi = []
        cum_aoi = 0.0

        for idx, r in enumerate(rows):
            time_idx = r["ep"] * frames_per_episode * num_slots + r["frame"] * num_slots + r["slot"]
            aoi_val = r.get("aoi", 0.0)
            cum_aoi += aoi_val
            avg = cum_aoi / (idx + 1)
            t.append(time_idx)
            avg_aoi.append(avg)

        max_time = max(max_time, max(t))
        plt.plot(t, avg_aoi, marker='o', linestyle='-', label=f"U{uid}")

    # Episode-aligned x-ticks
    tick_interval = frames_per_episode * num_slots
    episode_ticks = np.arange(0, num_episodes * tick_interval + 1, tick_interval)

    plt.xlabel("Slot Index")
    plt.ylabel("Avg AoI (Cumulative)")
    plt.title("Moving Average AoI per User")
    plt.xticks(episode_ticks)
    plt.xlim([0, max_time])
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, fontsize=7)
    plt.tight_layout()

    out_path = os.path.join(out_dir, out_pdf)
    plt.savefig(out_path, dpi=600, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved combined user AoI → {out_path}")

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

def compute_moving_avg_aoi_per_user(telemetry):
    import numpy as np
    avg_aoi_per_user = {}

    for uid, rows in telemetry.by_uid.items():
        rows = sorted(rows, key=lambda r: (r["ep"], r["frame"], r["slot"]))
        aois = [float(r["aoi"]) for r in rows]
        moving_avg = np.cumsum(aois) / np.arange(1, len(aois) + 1)
        avg_aoi_per_user[uid] = moving_avg

    return avg_aoi_per_user

def compute_avg_aoi_per_frame(telemetry, num_slots):
    import numpy as np, collections

    user_frame_aoi = collections.defaultdict(lambda: collections.defaultdict(list))

    for uid, rows in telemetry.by_uid.items():
        for r in rows:
            frame_key = (r["ep"], r["frame"])
            user_frame_aoi[uid][frame_key].append(float(r["aoi"]))

    avg_aoi_per_user_per_frame = {
        uid: {
            frame: np.mean(aoi_list)
            for frame, aoi_list in frame_dict.items()
        } for uid, frame_dict in user_frame_aoi.items()
    }

    # System-wide frame average (averaging over users per frame)
    system_avg_per_frame = {}
    all_frames = set()
    for frame_dict in avg_aoi_per_user_per_frame.values():
        all_frames.update(frame_dict.keys())

    for frame in sorted(all_frames):
        vals = [
            frame_dict[frame]
            for frame_dict in avg_aoi_per_user_per_frame.values()
            if frame in frame_dict
        ]
        if vals:
            system_avg_per_frame[frame] = np.mean(vals)

    return avg_aoi_per_user_per_frame, system_avg_per_frame

def compute_avg_aoi_per_episode(telemetry):
    import numpy as np, collections

    user_episode_aoi = collections.defaultdict(lambda: collections.defaultdict(list))

    for uid, rows in telemetry.by_uid.items():
        for r in rows:
            ep = r["ep"]
            user_episode_aoi[uid][ep].append(float(r["aoi"]))

    avg_aoi_per_user_per_ep = {
        uid: {
            ep: np.mean(aoi_list)
            for ep, aoi_list in ep_dict.items()
        } for uid, ep_dict in user_episode_aoi.items()
    }

    # System average over users for each episode
    system_avg_per_ep = {}
    all_eps = set()
    for ep_dict in avg_aoi_per_user_per_ep.values():
        all_eps.update(ep_dict.keys())

    for ep in sorted(all_eps):
        vals = [
            ep_dict[ep]
            for ep_dict in avg_aoi_per_user_per_ep.values()
            if ep in ep_dict
        ]
        if vals:
            system_avg_per_ep[ep] = np.mean(vals)

    return avg_aoi_per_user_per_ep, system_avg_per_ep

import os, tempfile, json, numpy as np

def flush_episode_safe(by_uid_dict, run_dir, M_total, num_slots, ep):
    """
    Save one episode chunk atomically:
      {run_dir}/chunks/slotwise_data_epXXXX_U{M}S{S}.npy
    Skips writing if there are no rows.
    """
    # Guard: no rows -> do not create an empty file
    total_rows = sum(len(v) for v in by_uid_dict.values())
    if total_rows == 0:
        print(f"[FLUSH] ep={ep} has no rows; skipping chunk write.")
        return None

    chunks_dir = os.path.join(run_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    out_path = os.path.join(chunks_dir, f"slotwise_data_ep{int(ep):04d}_U{M_total}S{num_slots}.npy")

    # Atomic write to local disk temp file, then replace.
    dstdir = os.path.dirname(out_path) or "."
    fd, tmp = tempfile.mkstemp(suffix=".tmp", dir=dstdir)
    try:
        # Write with numpy (close explicitly)
        with os.fdopen(fd, "wb") as f:
            #np.save(f, dict(by_uid_dict), allow_pickle=True)
            f.flush()
            os.fsync(f.fileno())
        # Replace atomically
        if os.path.exists(out_path):
            os.remove(out_path)
        os.replace(tmp, out_path)
    except Exception:
        try: os.remove(tmp)
        except: pass
        raise

    # Final sanity check: non-zero size
    try:
        sz = os.path.getsize(out_path)
        print(f"[FLUSH] ep={ep} → {out_path} ({sz} bytes, rows={total_rows})")
    except Exception:
        print(f"[FLUSH] ep={ep} → {out_path} (size unknown, rows={total_rows})")

    return out_path

import numpy as np
import matplotlib.pyplot as plt
import os

def load_episode_telemetry(run_dir, filename="episode_data.npy"):
    """
    Load telemetry data from a .npy file stored in a run directory.
    """
    import os
    path = os.path.join(run_dir, filename)
    print(f"[LOAD] Loading episode data from {path}")
    return np.load(path, allow_pickle=True).item()

import os, glob, numpy as np
from collections import defaultdict

def merge_chunks_safe(run_dir, M_total, num_slots, final_filename=None, delete_bad=False):
    """
    Merge all episode chunks into a single final file:
      slotwise_dataU{M}S{S}.npy
    Skips 0-byte and corrupted chunks. Returns the merged dict.
    """
    if final_filename is None:
        final_filename = f"slotwise_dataU{M_total}S{num_slots}.npy"
    final_path = os.path.join(run_dir, final_filename)

    chunks_dir = os.path.join(run_dir, "chunks")
    patt = os.path.join(chunks_dir, f"slotwise_data_ep*_U{M_total}S{num_slots}.npy")
    paths = sorted(glob.glob(patt))

    merged = defaultdict(list)
    good, bad = 0, 0

    for cp in paths:
        try:
            if not os.path.exists(cp) or os.path.getsize(cp) == 0:
                bad += 1
                print(f"[MERGE] skip empty: {os.path.basename(cp)}")
                if delete_bad:
                    try: os.remove(cp)
                    except: pass
                continue
            d = np.load(cp, allow_pickle=True).item()
            if not isinstance(d, dict):
                bad += 1
                print(f"[MERGE] skip non-dict: {os.path.basename(cp)}")
                if delete_bad:
                    try: os.remove(cp)
                    except: pass
                continue
            for k, v in d.items():
                merged[int(k)].extend(v)
            good += 1
        except Exception as e:
            bad += 1
            print(f"[MERGE] skip corrupt: {os.path.basename(cp)} ({e})")
            if delete_bad:
                try: os.remove(cp)
                except: pass

    total_rows = sum(len(v) for v in merged.values())
    if total_rows == 0:
        print(f"[FINAL] Nothing to write (good={good}, bad={bad}). Skipping final.")
        return dict(merged)

    # Atomic final save
    tmp = final_path + ".tmp"
    #np.save(tmp, dict(merged), allow_pickle=True)
    if os.path.exists(final_path):
        os.remove(final_path)
    os.replace(tmp, final_path)
    print(f"[FINAL] wrote {final_path} (uids={len(merged)}, rows={total_rows}, chunks: good={good}, bad={bad})")
    return dict(merged)


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

def plot_slotwise_rewards(sar_log_path, out_dir="telemetry_plots", window=10):
    import os, pickle
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt

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

    with open(sar_log_path, "rb") as f:
        sar_data = pickle.load(f)

    rewards = [entry["reward"] for entry in sar_data if "reward" in entry]
    steps = list(range(len(rewards)))

    # Moving average
    def moving_avg(data, k):
        return np.convolve(data, np.ones(k) / k, mode='valid')

    ma_rewards = moving_avg(rewards, window)
    ma_steps = steps[window - 1:]

    # Plot 1: raw slot-wise reward
    plt.figure(figsize=(6, 2.5))
    plt.plot(steps, rewards, color="tab:orange", label="Reward per slot")
    plt.xlabel("Slot Index")
    plt.ylabel("Reward")
    plt.title("Slot-wise Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_slotwise.pdf"), dpi=600)
    plt.close()

    # Plot 2: moving average reward
    plt.figure(figsize=(6, 2.5))
    plt.plot(ma_steps, ma_rewards, color="tab:purple", label=f"Moving Avg (k={window})")
    plt.xlabel("Slot Index")
    plt.ylabel("Avg Reward")
    plt.title(f"Moving Average Reward (window={window})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_moving_avg.pdf"), dpi=600)
    plt.close()
    print(f"[PLOT] Reward plots saved in {out_dir}")

policy = PPOActorCritic(state_dim, num_slots).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
buffer = RolloutBuffer(capacity=200000, state_dim=state_dim, n_slots=num_slots, device=device)

RUN_DIR = make_run_dir(M_total, num_slots)

try:
    import torch
except ImportError:
    torch = None

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


# 4. >>> SAVE META FILE HERE <<<
with open(os.path.join(RUN_DIR, "meta.json"), "w") as f:
    json.dump(run_meta, f, indent=2)

print(f"[SAVE] Run dir created: {RUN_DIR}")

settings_file = save_run_settings(
    out_dir=RUN_DIR,

    # --- Environment ---
    num_slots=num_slots,
    frames_per_episode=frames_per_episode,
    num_episodes=num_episodes,
    M_total=M_total,
    K_clusters=K_clusters,
    KF_clusters=KF_clusters,
    K_r_user=K_r_user,
    alpha_c=alpha_c, beta_c=beta_c,
    delta_wet=delta_wet, delta_wit=delta_wit,
    P_HAP=P_HAP, noise_pow=noise_pow,
    gamma_th_db=gamma_th_db,
    v_L=v_L, drop_prob=drop_prob, D=D, tau=tau, max_battery= battery_max,

    # --- PPO ---
    state_dim=state_dim, lr=lr, ppo_epochs=ppo_epochs,
    batch_size_I=batch_size_I, gamma_I=gamma_I, lam_I=lam_I,
    clip_range_I=clip_range_I, ent_coef_I=ent_coef_I,
    vf_coef_I=vf_coef_I, max_grad_norm_I=max_grad_norm_I,
    epsilon=epsilon, EPS=EPS,

    # --- Early stop ---
    PI_TARGET=PI_TARGET, PI_TOL=PI_TOL, PI_PATIENCE=PI_PATIENCE,
    SAVE_NAME=SAVE_NAME, EVAL_EPISODES=EVAL_EPISODES,

    # --- ε-Greedy schedule ---
    epsilon_start=epsilon_start, epsilon_final=epsilon_final,
    decay_steps=decay_steps, global_step=global_step
)


avg_aoi_hist = []
pi_loss_hist = []
v_loss_hist  = []
ent_hist     = []
kl_hist      = []

rng = np.random.default_rng(1234)
# before training loop
record_targets = {0, 1, 2}   # 0-based indices: record episodes 1 and 2
recorder = None


#aoi_avg_ep = np.zeros((num_episodes), dtype=float)
# --- init users ---

#users = [UserState(f"U{i+1}", d_init=np.random.uniform(1.0, D)) for i in range(M_total)]
users = []

# First half: near users, distance ∈ [1, D/2]
for i in range(M_total // 2):
    d_init = np.random.uniform(1.0, D / 2)
    users.append(UserState(f"U{i+1}", d_init=d_init, max_bat = battery_max))

# Second half: far users, distance ∈ [D/2 + 0.1, D]
for i in range(M_total // 2, M_total):
    d_init = np.random.uniform(D / 2 + 0.1, D)
    users.append(UserState(f"U{i+1}", d_init=d_init, max_bat = battery_max))

stop_counter = 0
stopped_at_ep = None
EV = 0
# replace: aoi_avg_ep = np.zeros((num_episodes), dtype=float)
aoi_avg_ep = []     # dynamic list: training + frozen eval will both append here
#stopped_at_ep = None
total_see_slot_EV = 0
reward_timeline = []
reward_per_frame = []  # list for all frames of all episodes
rewards_frame = []
Run_mode = "ppo"  # "ppo" | "greedy" | "random" | "threshold"
EP_Reward_sum = []
decision_timeline = []  # global list[DecisionLog]


# initial channel sample & AoI
for i, u in enumerate(users):
    u.sample_channel_and_gamma(K_r_user, alpha_c, beta_c)
    # u.aoi = 1
    # u.aoi_prev = 1
    # u.aoi_sum = 0.0
    # u.aoi_count = 0
    u.decode = 0
    u.succ_ema = 0.5
    u.last_decoded = 0
    u.last_slot_used = None
    u.assigned_this_frame = False
    # u.battery = 0.1
    # aoi_users[i, 0] = u.aoi

# initial grouping
near_all, far_all = split_groups_by_distance(users, tau)
allowed_near, idle_near, allowed_far, idle_far = init_allowed_idle(near_all, far_all, num_slots)
#telemetry = new_buffer()  # episode buffer
telemetry = Telemetry()

for ep in range(1, num_episodes + 1):

    for i, u in enumerate(users):
        #u.sample_channel_and_gamma(K_r_user, alpha_c, beta_c)
        # u.aoi = 1
        # u.aoi_prev = 1
        # u.aoi_sum = 0.0
        # u.aoi_count = 0
        u.decode = 0
        u.succ_ema = 0.5
        u.last_decoded = 0
        u.last_slot_used = None
        u.assigned_this_frame = False
        # u.battery = 0.1
        # aoi_users[i, 0] = u.aoi


    for frame in range(frames_per_episode):
        frame_reward_sum = 0.0  # accumulate rewards of all slots in this frame
        # --- sample channels for this frame, remember AoI_prev ---
        for u in users:
            u.gamma = u.sample_channel_and_gamma(K_r_user, alpha_c, beta_c)
            u.prev_frame_aoi_start = u.aoi
            u.decode_prev = u.decode #to store previous decoded state
            u.decode = 0
            E = u.compute_energy_harvested(u.gamma, delta_wet, P_HAP)
            u.harvested = E
            u.battery += E
            u.battery_check()
            u.last_slot_used = -1
            # --- make pairs & singles from allowed pools ---
        pairs, singles = make_pairs(allowed_near, allowed_far, num_slots)

        (assigned_slots, slot_map,
         states_to_buffer, masks_to_buffer, values_to_buffer, logps_to_buffer, actions_to_buffer,
         idx_by_slot, slot_mask) = assign_slots_for_frame(
            frame = frame,
            ep = ep,
            pairs=pairs,
            singles=singles,
            num_slots=num_slots,
            device=device,
            build_pair_state=build_pair_state,
            policy=policy,  # required for "ppo"
            epsilon_schedule=epsilon_schedule,  # your function, or None
            global_step=global_step,  # your counter, or None
            greedy_act=False,  # True => argmax for PPO if you want
            # threshold params:
            tau=10.0, alpha=1.0, beta=0.1,
            # greedy params:
            greedy_mode="sum",
            rng=random
        )

        scheduled_role = {}
        for (uH, uL) in pairs:
            uH.v_assigned_target = v_H
            uL.v_assigned_target = v_L
            scheduled_role[uH.uid] = 'H'
            scheduled_role[uL.uid] = 'L'

        for sl in range(num_slots):
            # everyone ages by +1
            for u in users:
                u.aoi += 1
                #u.decode = 0
                u.last_decoded_slot += 1
            scheduled = {}
            if sl in slot_map:
                entry = slot_map[sl]
                if entry.__class__ is tuple and len(entry) == 2 and isinstance(entry[0], UserState):
                    # ("pair", (uN,uF), s)
                    uN, uF = entry
                    dec_H, dec_L, sinr_H, sinr_L, battH, battL, reqH, reqL = decode_pair_pd_noma(
                        uN, uF, v_H, v_L, gamma_th, noise_pow, K_clusters,
                        alpha_c, beta_c, delta_wet, delta_wit, P_HAP
                    )
                    if dec_H:
                        uN.aoi = 1
                        uN.decode = 1
                        # distance refresh on success (example: re-sample)
                        uN.d = float(np.random.uniform(1.0, D))
                        uN.last_decoded_slot = 1
                    else:
                        uN.decode = 0
                    if dec_L:
                        uF.aoi = 1
                        uF.decode = 1
                        uF.d = float(np.random.uniform(1.0, D))
                        uF.last_decoded_slot = 1
                    else:
                        uF.decode = 0
                    #r_pair = -(uN.aoi + uF.aoi)
                    #rewards = r_pair
                    r_t = - (uN.aoi  + uF.aoi) #(sum(u.aoi for u in users) )  # negative mean AoI over all users
                    # example: reward = some_reward_value
                    sar_logger.set_reward(ep, frame, sl, r_t)
                    #reward_timeline.append(float(r_t))
                    frame_reward_sum += float(r_t)
                    # (Optional) write per-slot timeline for this frame


                    #scheduled[uN.uid] = dict(kind="NOMA-H", sinr=sinr_H, battery=battH,
                     #                        harvested=uN.harvested, required=reqH, decoded=int(dec_H))
                    #scheduled[uF.uid] = dict(kind="NOMA-L", sinr=sinr_L, battery=battL,
                      #                       harvested=uF.harvested, required=reqL, decoded=int(dec_L))

                    k = idx_by_slot.get(sl, 0)
                    if k is not None:
                        buffer.add(
                            states_to_buffer[k],
                            masks_to_buffer[k],
                            torch.tensor(actions_to_buffer[k], dtype=torch.long, device=device),
                            logps_to_buffer[k],
                            values_to_buffer[k],
                            r_t,  # scalar float is fine for most buffers
                            0.0  # not done within the frame
                        )
                    # pair (NOMA)
                    scheduled[uN.uid] = dict(kind="NOMA-H", sinr=sinr_H, battery=battH,
                                             harvested=uN.harvested, required=reqH, decoded=int(dec_H),
                                             scheduled=1, pd_role="NOMA-H")
                    scheduled[uF.uid] = dict(kind="NOMA-L", sinr=sinr_L, battery=battL,
                                             harvested=uF.harvested, required=reqL, decoded=int(dec_L),
                                             scheduled=1, pd_role="NOMA-L")
                    telemetry.log_user(ep=ep, u=uN, frame=frame, slot=sl, kind="PDNOMA", sinr=sinr_H, battery=uN.battery,
                                       harvested=uN.harvested, required=reqH, decoded=uN.decode, aoi=uN.aoi,distance=uN.d,
                                       scheduled=1, pd_role="NOMA-H")
                    telemetry.log_user(ep=ep, u=uF, frame=frame, slot=sl, kind="PDNOMA", sinr=sinr_L,
                                       battery=uF.battery,
                                       harvested=uF.harvested, required=reqL, decoded=uF.decode, aoi=uF.aoi,
                                       distance=uF.d,
                                       scheduled=1, pd_role="NOMA-L")

                else:
                    # ("single", u, s)
                    entry = slot_map[sl]
                    u_single = entry
                    dec, sinr, batt, req = decode_single_oma(u_single, v_H, gamma_th, noise_pow, K_clusters, alpha_c, beta_c, delta_wet, delta_wit, P_HAP)

                    if dec:
                        u_single.aoi = 1
                        u_single.decode = 1
                        u_single.d = float(np.random.uniform(1.0, D))
                        u_single.last_decoded_slot = 1
                    else:
                        u_single.decode = 0
                    #r_single = -u_single.aoi
                    #rewards = r_single
                    r_t = - u_single.aoi # (sum(u.aoi for u in users) )  # negative mean AoI over all users
                    sar_logger.set_reward(ep, frame, sl, r_t)
                    #reward_timeline.append(float(r_t))
                    frame_reward_sum += float(r_t)


                    #scheduled[u_single.uid] = dict(kind="OMA", sinr=sinr, battery=batt,
                     #                              harvested=u_single.harvested, required=req,
                      #                             decoded=int(dec))

                    k = idx_by_slot.get(sl, 0)
                    if k is not None:
                        buffer.add(
                            states_to_buffer[k],
                            masks_to_buffer[k],
                            torch.tensor(actions_to_buffer[k], dtype=torch.long, device=device),
                            logps_to_buffer[k],
                            values_to_buffer[k],
                            r_t,  # scalar float is fine for most buffers
                            0.0  # not done within the frame
                        )


                    scheduled[u_single.uid] = dict(kind="OMA", sinr=sinr, battery=batt,
                                                   harvested=u_single.harvested, required=req, decoded=int(dec),
                                                   scheduled=1, pd_role="OMA")
                    telemetry.log_user(ep=ep, u=u_single, frame=frame, slot=sl, kind="OMA", sinr=sinr,
                                       battery=u_single.battery, harvested=u_single.harvested, required=req,
                                       decoded=u_single.decode, aoi=u_single.aoi,
                                       distance=u_single.d, scheduled=1, pd_role="OMA")

                global_step += 1
                # Fetch the right transition by slot -> index
                for i, u in enumerate(users):
                    u.update_belief(sl, 0.01)
                    info = scheduled.get(u.uid)
                    if info is None:
                        telemetry.log_user(ep=ep,
                        u=u,
                        frame=frame,
                        slot=sl,
                        kind="Idle",
                        sinr= 0,
                        battery=u.battery,
                        harvested=u.harvested,
                        required=0,
                        decoded=u.decode,
                        aoi=u.aoi,
                        distance=u.d,
                        scheduled=0,
                        pd_role="Idle"  # string
                        )
            for u in users:
                u.update_belief_energy(sl)

        # AoI at the end of this frame
        for u in users:
            u.prev_frame_aoi_end = u.aoi
            u.delta_aoi_prev = u.prev_frame_aoi_start - u.prev_frame_aoi_end

        # Update decoding success rate based on feedback
        for u in users:
            u.update_belief_psuccess(decoded=u.decode)

        succ_near = [e[1][0] for e in assigned_slots if e[0] == "pair" and e[1][0].decode == 1]
        succ_far = [e[1][1] for e in assigned_slots if e[0] == "pair" and e[1][1].decode == 1]
        # singles counted as near successes arbitrarily if they succeeded (aoi==1) and were formerly near
        succ_singles = [e[1] for e in assigned_slots if e[0] == "single" and e[1].decode == 1]
        for us in succ_singles:
            if us.d <= tau:
                succ_near.append(us)
            else:
                succ_far.append(us)

        # then drop/add from successes
        users_by_uid = {u.uid: u for u in users}  # users = list[UserState]
        allowed_near, idle_near = update_group_after_frame(allowed_near, idle_near,
                                                           succ_near, drop_prob, rng,
                                                           users_by_uid=users_by_uid)

        # update_group_after_frame(allowed_near, idle_near, succ_near, drop_prob, rng)
        allowed_far, idle_far = update_group_after_frame(allowed_far, idle_far,
                                                         succ_far, drop_prob, rng,
                                                         users_by_uid=users_by_uid)
        # update_group_after_frame(allowed_far,  idle_far,  succ_far,  drop_prob, rng)
        # --- update groups: reconcile after distance changes (success only) ---
        allowed_near, idle_near, allowed_far, idle_far = reconcile_after_successes(users, allowed_near,
                                                                                   idle_near, allowed_far,
                                                                                   idle_far, tau)

    system_avg = compute_average_aoi(telemetry, num_slots, frames_per_episode, RUN_DIR)
    print(f"[Episode {ep + 1}] System Avg AoI: {system_avg:.3f}")

    if ep % 1 == 0 or ep == 1:
        print(f"Ep {ep:4d}/{num_episodes}" ) #| AvgAoI={aoi_avg_ep[-1]:6.3f} | "
              #f"π_loss={stats['pi_loss']:.4f} | V_loss={stats['v_loss']:.4f} | "
              #f"Ent={stats['ent']:.4f} | KL={stats['kl']:.4f}")

    path = flush_episode_safe(telemetry.by_uid, RUN_DIR, M_total, num_slots, ep)
    telemetry.by_uid.clear()  # free RAM for next episode

    #telemetry.save_episode_npy(RUN_DIR, filename=f"slotwise_dataU{M_total}S{num_slots}.npy")
    #telemetry.save_slotwise_npy(RUN_DIR, filename=f"slotwise_dataU{M_total}S{num_slots}.npy")



#episode_data = np.load(os.path.join(RUN_DIR, "episode_data.npy"), allow_pickle=True).item()
# after the last episode:
#telemetry.finalize_run(
 #   RUN_DIR, M_total, num_slots,
 #   final_filename=f"slotwise_dataU{M_total}S{num_slots}.npy",
 #   keep_chunks=True  # set False to delete chunks after merging
#)

merged = merge_chunks_safe(RUN_DIR, M_total, num_slots,
                           final_filename=f"slotwise_dataU{M_total}S{num_slots}.npy",
                           delete_bad=False)

# Now compute metrics from 'merged' (dict[uid] -> list[rows])
per_user_moving = compute_moving_avg_aoi_per_user(merged)
user_frame_aoi, system_frame_aoi = compute_avg_aoi_per_frame(merged, num_slots)
user_ep_aoi, system_ep_aoi = compute_avg_aoi_per_episode(merged)


#per_user_moving = compute_moving_avg_aoi_per_user(telemetry)
#user_frame_aoi, system_frame_aoi = compute_avg_aoi_per_frame(telemetry, num_slots)
#user_ep_aoi, system_ep_aoi = compute_avg_aoi_per_episode(telemetry)

sar_logger.save(RUN_DIR, filename=f"sar_logU{M_total}S{num_slots}.pkl")

# Example print:
for ep, avg in system_ep_aoi.items():
    print(f"Episode {ep}: System Avg AoI = {avg:.3f}")

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

#plot_slotwise_rewards(RUN_DIR, out_dir=RUN_DIR, window=10)

plot_time_averaged_system_aoi(data, num_slots, frames_per_episode, RUN_DIR, out_pdf="system_aoi_time_avg.pdf")

#plot_all_users_aoi(telemetry, num_slots, frames_per_episode,  out_pdf="AOI_All_Users.pdf", out_dir=RUN_DIR)
#plot_all_users_energy(telemetry, num_slots, frames_per_episode, out_pdf="Energy_All_Users.pdf", out_dir=RUN_DIR)
