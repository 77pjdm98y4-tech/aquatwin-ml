"""
Generate a labeled, 1 Hz simulated AquaTwin dataset.

    python simulate.py                 # default: 10 runs x 3 h = 30 h
    python simulate.py --runs 10 --hours 2 --out data

Writes
    data/sensor_data.csv   one row per second: commands, sensor readings, labels
    data/events.csv        one row per injected fault (type, onset, end, details)

Runs 0-3 are normal operation (model training), runs 4-5 are normal (alarm
threshold calibration), runs 6+ contain the six induced faults from Interim
Report 3.2, each twice.
Same split idea applies to the real bench-top data later.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from physics import TestbedParams, run_engine

FAULT_TYPES = ["turbidity_spike", "pump_slowdown", "partial_blockage",
               "leak", "low_do", "sensor_drift"]

# sensor name -> (noise std, resolution)
SENSORS = {
    "flow_Lpm": (0.10, 0.01),
    "press_kPa": (0.15, 0.01),
    "dp_filter_kPa": (0.10, 0.01),
    "level_cm": (0.15, 0.1),
    "turb_in_NTU": (0.25, 0.01),
    "turb_out_NTU": (0.08, 0.01),
    "do_mgL": (0.04, 0.01),
    "temp_C": (0.03, 0.01),
    "ph": (0.01, 0.001),
    "cond_uScm": (1.5, 0.1),
}
COMMANDS = ["pump_speed_cmd", "valve_open_cmd", "aeration_cmd", "dose_NTU", "backwash", "refill"]


def make_schedule(n_runs, run_s, rng):
    rows = []
    for r in range(n_runs):
        t = np.arange(run_s)
        speed = np.empty(run_s); valve = np.empty(run_s); aer = np.empty(run_s, dtype=int)
        k = 0
        while k < run_s:
            seg = int(rng.uniform(1200, 2400))
            speed[k:k + seg] = rng.choice([0.7, 0.85, 1.0])
            valve[k:k + seg] = rng.choice([1.0, 1.0, 0.7, 0.5])
            aer[k:k + seg] = rng.choice([0, 1, 2, 3])
            k += seg
        dose = np.zeros(run_s)
        for _ in range(rng.integers(1, 3)):  # planned, logged kaolin doses
            t0 = int(rng.uniform(600, run_s - 1200)); amount = rng.uniform(10, 30)
            dose[t0:t0 + 30] = amount / 30
        df = pd.DataFrame(dict(run_id=r, t_s=t, pump_speed_cmd=speed, valve_open_cmd=valve,
                               aeration_cmd=aer, dose_NTU=dose, backwash=0, refill=0))
        df.loc[0, "backwash"] = 1
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def inject_faults(sched, first_fault_run, run_s, rng):
    """Place 3 faults per fault run, each type twice, at non-overlapping slots."""
    n = len(sched)
    f = pd.DataFrame(dict(f_speed_mult=np.ones(n), f_filter_mult=np.ones(n),
                          f_leak_coef=np.zeros(n), f_kla_mult=np.ones(n), f_dose_NTU=np.zeros(n)))
    fault_runs = sorted(sched.run_id.unique())[first_fault_run:]
    pool = FAULT_TYPES * int(np.ceil(3 * len(fault_runs) / len(FAULT_TYPES)))
    rng.shuffle(pool)
    drift_targets = iter(["ph", "do_mgL", "cond_uScm", "turb_out_NTU"] * 3)
    events, drifts = [], []
    label = np.array(["normal"] * n, dtype=object)
    fid = np.full(n, -1)
    slots = [int(run_s * x) for x in (0.15, 0.45, 0.75)]
    eid = 0
    for r in fault_runs:
        base = sched.index[sched.run_id == r][0]
        for s in slots:
            ftype = pool.pop()
            onset = base + s + int(rng.uniform(-300, 300))
            info = {}
            if ftype == "turbidity_spike":      # unlogged kaolin / sediment slug
                dur = 1200; amt = rng.uniform(15, 40)
                f.loc[onset:onset + 29, "f_dose_NTU"] = amt / 30; info = dict(added_NTU=round(amt, 1))
            elif ftype == "pump_slowdown":      # actual speed below command
                dur = 900; m = rng.uniform(0.75, 0.88)
                f.loc[onset:onset + dur - 1, "f_speed_mult"] = m; info = dict(speed_mult=round(m, 2))
            elif ftype == "partial_blockage":   # fouled filter media; backwash at end
                dur = 900; m = rng.uniform(1.8, 3.0)
                ramp = np.linspace(1, m, 120)
                f.loc[onset:onset + 119, "f_filter_mult"] = ramp
                f.loc[onset + 120:onset + dur - 1, "f_filter_mult"] = m
                sched.loc[onset + dur, "backwash"] = 1; info = dict(filter_mult=round(m, 2))
            elif ftype == "leak":               # leak downstream of pump; refill at end
                dur = 600; c = rng.uniform(3e-6, 6e-6)
                f.loc[onset:onset + dur - 1, "f_leak_coef"] = c
                sched.loc[onset + dur, "refill"] = 1; info = dict(emitter_coef=c)
            elif ftype == "low_do":             # air pump fails while commanded on
                dur = 900
                f.loc[onset:onset + dur - 1, "f_kla_mult"] = 0.0
                sched.loc[onset:onset + dur - 1, "aeration_cmd"] = np.maximum(
                    sched.loc[onset:onset + dur - 1, "aeration_cmd"], 2)
                info = {}
            elif ftype == "sensor_drift":       # handled at the sensor layer
                dur = 1200; tgt = next(drift_targets)
                total = {"ph": 0.35, "do_mgL": -1.2, "cond_uScm": 25.0, "turb_out_NTU": 1.5}[tgt]
                drifts.append((onset, dur, tgt, total)); info = dict(sensor=tgt, total_drift=total)
            label[onset:onset + dur] = ftype
            fid[onset:onset + dur] = eid
            events.append(dict(event_id=eid, run_id=r, fault_type=ftype,
                               onset_idx=onset, end_idx=onset + dur,
                               onset_t_s=int(sched.t_s[onset]), duration_s=dur, **{"details": info}))
            eid += 1
    return f, label, fid, pd.DataFrame(events), drifts


def add_sensor_layer(truth, sched, drifts, rng):
    n = len(sched)
    meas = {}
    run = sched.run_id.to_numpy()
    for name, (sd, res) in SENSORS.items():
        x = truth[name].copy()
        # per-run calibration offset (small, inside calibration tolerance)
        bias = rng.normal(0, sd * 0.5, size=run.max() + 1)[run]
        x = x + bias + rng.normal(0, sd, n)
        if name.startswith("turb"):  # occasional air-bubble spikes
            spikes = rng.random(n) < 0.0005
            x[spikes] += rng.uniform(1, 4, spikes.sum())
        meas[name] = x
    for onset, dur, tgt, total in drifts:
        meas[tgt][onset:onset + dur] += np.linspace(0, total, dur)
    for name, (sd, res) in SENSORS.items():
        x = np.round(meas[name] / res) * res
        if name not in ("temp_C",):
            x = np.maximum(x, 0)
        drop = rng.random(n) < 0.001  # random dropouts -> NaN
        x[drop] = np.nan
        meas[name] = x
    return meas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--normal-runs", type=int, default=6, help="runs without faults (train + calibration)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    run_s = int(a.hours * 3600)
    p = TestbedParams()
    sched = make_schedule(a.runs, run_s, rng)
    faults, label, fid, events, drifts = inject_faults(sched, a.normal_runs, run_s, rng)
    print(f"Simulating {a.runs} runs x {a.hours} h = {len(sched):,} s ...", flush=True)
    truth = run_engine(sched, p, faults=faults, seed=a.seed)
    meas = add_sensor_layer(truth, sched, drifts, rng)

    df = sched.copy()
    start = pd.Timestamp("2026-11-11 09:00:00")
    df.insert(0, "timestamp", [start + pd.Timedelta(days=int(r), seconds=int(t))
                               for r, t in zip(df.run_id, df.t_s)])
    for k, v in meas.items():
        df[k] = v
    df["fault_type"] = label
    df["fault_id"] = fid
    n_norm_train = a.normal_runs - 2 if a.normal_runs >= 4 else max(a.normal_runs - 1, 1)
    df["split"] = np.where(df.run_id < n_norm_train, "train",
                   np.where(df.run_id < a.normal_runs, "calibration", "test"))
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "sensor_data.csv"), index=False)
    events.to_csv(os.path.join(a.out, "events.csv"), index=False)
    print(f"Wrote {len(df):,} rows and {len(events)} fault events to {a.out}/")
    print(events[["event_id", "run_id", "fault_type", "onset_t_s", "duration_s", "details"]].to_string(index=False))


if __name__ == "__main__":
    main()
