# PPO-Based PD-NOMA Scheduling for Age of Information Minimization

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO-purple)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](https://arxiv.org/)

---

## Overview

This repository contains the full implementation of the framework presented in:

> **"A PPO-Based PD-NOMA Scheduling Methodology for Age of Information
> Minimization"**
> Muhammad Tauseef Mushtaq, Nicola Cordeschi,
> Luigi Alfredo Grieco, and Gennaro Boggia
> Department of Electrical and Information Engineering,
> Politecnico di Bari, Bari, Italy

We address the problem of uplink scheduling in massive Machine-Type
Communication (mMTC) IoT networks, where energy-harvesting devices must
transmit status updates under dynamic channel conditions, inter-cell
interference, and strict battery constraints. The scheduling policy is
modeled as a **Partially Observable Markov Decision Process (POMDP)**
and solved using **Proximal Policy Optimization (PPO)**.

The framework combines:
- A **two-level PD-NOMA** scheme pairing near (high-power) and far
  (low-power) devices in every slot, enabling Successive Interference
  Cancellation (SIC) decoding.
- A **Drop-and-Add pool management algorithm** (Algorithm 1) that
  refreshes the active user set each frame based on decoding success
  and AoI priority.
- A **PPO-based slot assignment policy** (Algorithm 2) that learns
  to minimize the Average Age of Information (AAoI) under partial
  observability.

### Key Results

| Metric | PPO | Threshold | Greedy | Random |
|--------|-----|-----------|--------|--------|
| Avg AoI @ M=60, γ_th=−5 dB | **~80 slots** | ~240 slots | ~380 slots | ~480 slots |
| Avg AoI @ M=75, γ_th=−5 dB | **~90 slots** | ~340 slots | ~430 slots | ~595 slots |
| AoI increase (M=60→75) | **12.54%** | ~41% | ~13% | ~24% |
| AoI @ γ_th = 0 dB | **~108 slots** | ~280 slots | ~420 slots | ~530 slots |

> The PPO agent maintains a **4× AoI increase** against a
> **20 dB increase** in SINR decoding threshold, demonstrating
> robust learning of physical-layer uncertainty.

---

## Key Features

- **PD-NOMA with SIC decoding**: near/far user grouping by distance
  threshold τ; power levels v_H and v_L designed to satisfy SIC SINR
  constraints; inter-cell interference modeled explicitly.

- **Non-linear energy harvesting**: devices recharge via Wireless Energy
  Transfer (WET) in all slots except their assigned transmission slot;
  battery state gates actual transmission eligibility.

- **Drop-and-Add user pool (Algorithm 1)**: after each frame, successfully
  decoded devices are rotated out and replaced by the highest-AoI waiting
  devices, guaranteeing fairness and freshness across all M users.

- **POMDP with partial observability**: the agent observes per-pair AoI,
  AoI delta, distance, energy belief, success probability, scheduling
  recency, and a global success-history window — without direct access
  to channel transition probabilities.

- **Per-pair PPO action**: for each near/far user pair occupying one
  slot, the agent assigns a slot index; the full frame assignment vector
  f_k^i is built sequentially, one pair at a time.

- **Four-policy benchmark**: PPO vs. Threshold-based, Greedy (highest
  AoI first), and Random scheduling — evaluated across traffic loads
  M ∈ {60, 65, 70, 75} and SINR thresholds
  γ_th ∈ {−20, −10, −5, 0} dB.

---

## Repository Structure
