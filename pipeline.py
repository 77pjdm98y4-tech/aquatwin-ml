"""
AquaTwin ML pipeline: digital-twin residuals -> anomaly detection + prediction.

    python pipeline.py                      # uses data/sensor_data.csv
    python pipeline.py --data data --out results

Steps
  1. Data quality: fill short dropouts, flag them.
  2. Digital twin: run the physics engine (no faults, slightly-off params)
     on the logged commands -> expected value for every sensor. The twin
     tracks the filter's clogging state from measured head loss with a
     30-minute time constant (simple data assimilation).
  3. Features: residuals (measured - twin), trailing rolling means,
     slopes (for drift) and short-window variability. Trailing windows only,
     so every feature is computable in real time.
  4. Anomaly detection: Isolation Forest (primary), LOF (validation),
     residual z-score control chart (baseline). Trained on normal runs;
     alarm threshold set on a separate normal calibration run; a
     persistence rule (k of n seconds) suppresses single-sample blips.
  5. Prediction (hybrid twin): Random Forest and XGBoost learn the gap
     between physics and measurement for outlet turbidity, filter-inlet
     pressure and filter head loss, from commands + physics outputs only.
     Compared against the physics-only prediction.
  6. Evaluation: sample-level confusion matrix, event-level detection rate
     and delay, false alarms per hour, R2 / MAPE, streaming latency.

Outputs in results/: metrics.json, report.md, figures, saved models.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from physics import TestbedParams, run_engine, twin_params

SENSORS = ["flow_Lpm", "press_kPa", "dp_filter_kPa", "level_cm", "turb_in_NTU",
           "turb_out_NTU", "do_mgL", "ph", "cond_uScm"]
ALL_SENSORS = SENSORS + ["temp_C"]
COMMANDS = ["pump_speed_cmd", "valve_open_cmd", "aeration_cmd", "dose_NTU", "backwash", "refill"]
SLOPE_SENSORS = ["ph", "do_mgL", "cond_uScm", "turb_out_NTU", "level_cm"]
WARMUP_S = 300        # rolling windows need history; skip first 5 min of each run
GRACE_S = 900         # after a fault ends the process takes time to recover; not scored
PERSIST_N, PERSIST_K = 20, 15   # alarm if >= 15 of the last 20 s are flagged
TARGET_FPR = 0.005    # per-sample flag rate allowed on the calibration run
PRED_TARGETS = ["turb_out_NTU", "press_kPa", "dp_filter_kPa"]


# ---------------------------------------------------------------- 1. data quality
def clean(df):
    df = df.copy()
    df["dq_missing"] = df[ALL_SENSORS].isna().sum(axis=1)
    df[ALL_SENSORS] = (df.groupby("run_id")[ALL_SENSORS]
                         .transform(lambda s: s.interpolate(limit=10, limit_direction="both")))
    return df


# ---------------------------------------------------------------- 2. digital twin
def add_twin(df, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        tw = pd.read_csv(cache_path)
        if len(tw) == len(df):
            for c in tw.columns:
                df[c] = tw[c].to_numpy()
            return df
    sched = df[["run_id", "t_s"] + COMMANDS].copy()
    pred = run_engine(sched, twin_params(TestbedParams()), faults=None,
                      measured_temp=df["temp_C"].to_numpy(),
                      measured_dp=df["dp_filter_kPa"].to_numpy(),
                      measured_flow=df["flow_Lpm"].to_numpy())
    tw = pd.DataFrame({f"twin_{c}": pred[c] for c in SENSORS})
    if cache_path:
        tw.to_csv(cache_path, index=False)
    for c in tw.columns:
        df[c] = tw[c].to_numpy()
    return df


# ---------------------------------------------------------------- 3. features
def _rolling_slope(y, w):
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    mt, my = t.rolling(w, min_periods=10).mean(), y.rolling(w, min_periods=10).mean()
    mty = (t * y).rolling(w, min_periods=10).mean()
    mtt = (t * t).rolling(w, min_periods=10).mean()
    return (mty - mt * my) / (mtt - mt * mt)


def run_features(g):
    """Features for ONE run (or one streaming buffer). Trailing windows only."""
    X = {}
    for c in SENSORS:
        r = g[c] - g[f"twin_{c}"]
        X[f"r_{c}"] = r
        X[f"r_{c}_m30"] = r.rolling(30, min_periods=1).mean()
        X[f"r_{c}_m120"] = r.rolling(120, min_periods=1).mean()
    for c in SLOPE_SENSORS:
        X[f"r_{c}_slope300"] = _rolling_slope(g[c] - g[f"twin_{c}"], 300) * 60  # units/min
    for c in ["flow_Lpm", "press_kPa", "turb_in_NTU"]:
        X[f"{c}_sd30"] = g[c].rolling(30, min_periods=2).std()
    removal = 1 - g["turb_out_NTU"] / g["turb_in_NTU"].clip(lower=0.1)
    twin_removal = 1 - g["twin_turb_out_NTU"] / g["twin_turb_in_NTU"].clip(lower=0.1)
    X["r_removal_m60"] = (removal - twin_removal).rolling(60, min_periods=1).mean()
    return pd.DataFrame(X, index=g.index)


def build_features(df):
    X = pd.concat([run_features(g) for _, g in df.groupby("run_id")]).loc[df.index]
    return X.fillna(0.0)


def persistence(flags, run_id):
    s = pd.Series(flags.astype(int))
    return (s.groupby(run_id).transform(lambda x: x.rolling(PERSIST_N, min_periods=1).sum())
              .to_numpy() >= PERSIST_K)


# ---------------------------------------------------------------- 4. anomaly detection
class Detector:
    """Wraps scaler + scorer + calibrated threshold. Higher score = more anomalous."""

    def __init__(self, kind, features):
        self.kind, self.features = kind, features
        self.scaler = StandardScaler()

    def fit(self, X):
        Z = self.scaler.fit_transform(X[self.features])
        if self.kind == "isolation_forest":
            self.model = IsolationForest(n_estimators=200, max_samples=1024, random_state=42, n_jobs=-1).fit(Z)
        elif self.kind == "lof":
            sub = Z[np.random.default_rng(0).choice(len(Z), min(6000, len(Z)), replace=False)]
            self.model = LocalOutlierFactor(n_neighbors=50, novelty=True).fit(sub)
        elif self.kind == "zscore":
            self.model = None
        return self

    def score(self, X):
        Z = self.scaler.transform(X[self.features])
        if self.kind == "zscore":
            return np.abs(Z).max(axis=1)
        return -self.model.score_samples(Z)

    def calibrate(self, X_cal):
        self.threshold = float(np.quantile(self.score(X_cal), 1 - TARGET_FPR))
        return self

    def alarms(self, X, run_id):
        return persistence(self.score(X) > self.threshold, run_id)


def scoring_mask(df, events):
    """True where a sample is scored: past warm-up and not in post-fault recovery."""
    m = df["t_s"].to_numpy() >= WARMUP_S
    for _, e in events.iterrows():
        m[e.end_idx:e.end_idx + GRACE_S] = False
    return m


def evaluate_detector(alarm, df, events, mask):
    y = (df["fault_type"] != "normal").to_numpy()
    tn, fp, fn, tp = confusion_matrix(y[mask], alarm[mask], labels=[False, True]).ravel()
    ev_rows = []
    for _, e in events.iterrows():
        win = alarm[e.onset_idx:e.end_idx]
        hit = bool(win.any())
        ev_rows.append(dict(event_id=int(e.event_id), fault_type=e.fault_type, detected=hit,
                            delay_s=int(np.argmax(win)) if hit else None,
                            alarm_already_on=bool(alarm[e.onset_idx - 1]) if e.onset_idx > 0 else False))
    ev = pd.DataFrame(ev_rows)
    # false-alarm episodes: rising edges of the alarm inside scored normal samples
    normal_scored = mask & ~y
    edges = np.flatnonzero(np.diff(np.r_[0, alarm.astype(int)]) == 1)
    fa = int(sum(normal_scored[i] for i in edges))
    hours_normal = normal_scored.sum() / 3600
    return dict(
        sample_level=dict(tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                          sensitivity=round(tp / max(tp + fn, 1), 3),
                          specificity=round(tn / max(tn + fp, 1), 3),
                          precision=round(tp / max(tp + fp, 1), 3)),
        event_level=dict(detection_rate=round(ev.detected.mean(), 3),
                         median_delay_s=float(ev.delay_s.dropna().median()) if ev.detected.any() else None,
                         false_alarms_per_hour=round(fa / max(hours_normal, 1e-9), 2)),
        per_event=ev,
    )


# ---------------------------------------------------------------- 5. prediction
def prediction_frame(df):
    """Inputs a hybrid twin is allowed to use: operator commands, measured
    temperature, time since backwash, recent planned doses, and the physics
    twin's own outputs. NO other measured sensors, so the prediction is a true
    'what should this sensor read right now' estimate (also usable as a virtual
    sensor if a probe fails)."""
    g = df.groupby("run_id")
    P = df[["pump_speed_cmd", "valve_open_cmd", "aeration_cmd", "temp_C"]
           + [f"twin_{c}" for c in SENSORS]].copy()
    bw_block = g["backwash"].cumsum()
    P["s_since_backwash"] = df.groupby([df.run_id, bw_block]).cumcount()
    P["dose_last600"] = g["dose_NTU"].transform(lambda s: s.rolling(600, min_periods=1).sum())
    return P


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / np.clip(np.abs(y), 1e-6, None))) * 100)


def fit_predictors(df, train_mask, test_mask, normal_mask, target):
    """Hybrid twin: physics prediction + ML correction. The model learns the
    residual (measured - physics) from normal data; prediction = physics + ML."""
    P = prediction_frame(df)
    resid = df[target] - df[f"twin_{target}"]
    models = {"random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=20,
                                                      max_features=0.5, n_jobs=-1, random_state=42)}
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    except ImportError:
        pass
    fitted = {}
    for name, m in models.items():
        m.fit(P[train_mask], resid[train_mask]); fitted[name] = m

    def score(mask):
        yt, phys = df.loc[mask, target].to_numpy(), df.loc[mask, f"twin_{target}"].to_numpy()
        preds = {"physics twin only": phys}
        for name, m in fitted.items():
            preds[f"hybrid: physics + {name}"] = phys + m.predict(P[mask])
        return {n: dict(r2=round(r2_score(yt, yh), 3), mape=round(mape(yt, yh), 2),
                        accuracy_pct=round(100 - mape(yt, yh), 1),
                        mae=round(mean_absolute_error(yt, yh), 3)) for n, yh in preds.items()}

    res = {"normal operation": score(test_mask & normal_mask), "all test samples": score(test_mask)}
    return res, fitted, list(P.columns)


# ---------------------------------------------------------------- 6. streaming latency
class StreamingTwin:
    """What runs on the lab server: keep a 300 s buffer, compute features for the
    newest reading, score it. The physics step is cheap because EPANET results
    are cached per operating point; a cache miss costs one EPANET solve."""

    def __init__(self, detector, buffer_s=300):
        self.det = detector
        self.buf = deque(maxlen=buffer_s)
        self.recent = deque(maxlen=PERSIST_N)

    def step(self, row: dict):
        self.buf.append(row)
        g = pd.DataFrame(list(self.buf))
        x = run_features(g).iloc[[-1]].fillna(0.0)
        s = float(self.det.score(x)[0])
        self.recent.append(s > self.det.threshold)
        return s, sum(self.recent) >= PERSIST_K


def latency_benchmark(df, det, n=1500):
    st = StreamingTwin(det)
    cols = SENSORS + [f"twin_{c}" for c in SENSORS]
    rows = df[cols].iloc[:n].to_dict("records")
    times = []
    for r in rows:
        t0 = time.perf_counter(); st.step(r); times.append(time.perf_counter() - t0)
    times = np.array(times[300:]) * 1000  # skip buffer fill
    from physics import solve_hydraulics
    p = TestbedParams()
    t0 = time.perf_counter(); solve_hydraulics(p, 0.777, 0.83, 1.37); epa = (time.perf_counter() - t0) * 1000
    return dict(features_plus_score_ms_median=round(float(np.median(times)), 2),
                features_plus_score_ms_p95=round(float(np.percentile(times, 95)), 2),
                epanet_uncached_solve_ms=round(epa, 1))


# ---------------------------------------------------------------- figures
COLORS = dict(meas="#2a78d6", twin="#52514e", alarm="#d03b3b", fault="#fab219")


def plot_run(df, score, thr, alarm, events, run, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = df[df.run_id == run]; idx = g.index
    tmin = g.t_s / 60
    panels = [("flow_Lpm", "Flow (L/min)"), ("dp_filter_kPa", "Filter ΔP (kPa)"),
              ("level_cm", "Tank level (cm)"), ("turb_in_NTU", "Turbidity in (NTU)"),
              ("do_mgL", "DO (mg/L)"), ("ph", "pH")]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(11, 12), sharex=True,
                             gridspec_kw=dict(hspace=0.15))
    ev = events[events.run_id == run]
    for ax, (c, lab) in zip(axes, panels):
        ax.plot(tmin, g[c], lw=0.8, color=COLORS["meas"], label="measured")
        ax.plot(tmin, g[f"twin_{c}"], lw=1.6, color=COLORS["twin"], label="digital twin")
        ax.set_ylabel(lab, fontsize=9)
    ax = axes[-1]
    ax.plot(tmin, score[idx], lw=0.8, color=COLORS["meas"], label="anomaly score")
    ax.axhline(thr, color=COLORS["twin"], lw=1, ls="--", label="threshold")
    a = alarm[idx]
    ax.fill_between(tmin, 0, 1, where=a, transform=ax.get_xaxis_transform(),
                    color=COLORS["alarm"], alpha=0.25, lw=0, label="alarm")
    ax.set_ylabel("Isolation Forest\nscore", fontsize=9); ax.set_xlabel("Time in run (min)")
    for axx in axes:
        for _, e in ev.iterrows():
            axx.axvspan(e.onset_t_s / 60, (e.onset_t_s + e.duration_s) / 60,
                        color=COLORS["fault"], alpha=0.18, lw=0)
        axx.spines[["top", "right"]].set_visible(False)
        axx.grid(axis="y", color="#e6e5e1", lw=0.6); axx.tick_params(labelsize=8)
    for _, e in ev.iterrows():
        axes[0].text((e.onset_t_s + e.duration_s / 2) / 60, 1.02, e.fault_type.replace("_", " "),
                     transform=axes[0].get_xaxis_transform(), ha="center", fontsize=8, color="#0b0b0b")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.35), ncol=2, frameon=False, fontsize=8)
    axes[-1].legend(loc="upper right", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Run {run}: measured vs digital twin, injected faults shaded", y=0.94, fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(os.path.join(a.data, "sensor_data.csv"))
    events = pd.read_csv(os.path.join(a.data, "events.csv"))
    df = clean(df)
    print("Running digital twin ...", flush=True)
    df = add_twin(df, cache_path=os.path.join(a.data, "twin_predictions.csv"))
    print("Building features ...", flush=True)
    X = build_features(df)
    feats = list(X.columns)

    split = df["split"].to_numpy()
    warm = df["t_s"].to_numpy() >= WARMUP_S
    tr, cal, te = (split == "train") & warm, (split == "calibration") & warm, split == "test"
    test_df = df[te].reset_index(drop=True)
    test_events = events.copy()
    offset = np.flatnonzero(te)[0]
    test_events["onset_idx"] -= offset; test_events["end_idx"] -= offset
    mask = scoring_mask(test_df, test_events)

    metrics = dict(data=dict(rows=len(df), hours=round(len(df) / 3600, 1),
                             train_hours=round(tr.sum() / 3600, 1),
                             calibration_hours=round(cal.sum() / 3600, 1),
                             test_hours=round(te.sum() / 3600, 1),
                             fault_events=len(events),
                             missing_readings_filled=int(df.dq_missing.sum())))
    detectors, per_event_tables = {}, {}
    zscore_feats = [f for f in feats if f.endswith("_m120") or f.endswith("_slope300")]
    smooth_feats = [f for f in feats if f.endswith(("_m30", "_m120", "_slope300", "_m60"))]
    for kind, fs in [("isolation_forest", smooth_feats), ("lof", smooth_feats), ("zscore", zscore_feats)]:
        print(f"Anomaly detector: {kind} ...", flush=True)
        det = Detector(kind, fs).fit(X[tr]).calibrate(X[cal])
        alarm = det.alarms(X[te].reset_index(drop=True), test_df.run_id.to_numpy())
        r = evaluate_detector(alarm, test_df, test_events, mask)
        per_event_tables[kind] = r.pop("per_event").drop(columns=[] if kind == "isolation_forest" else ["alarm_already_on"])
        metrics[kind] = r
        detectors[kind] = det
        if kind == "isolation_forest":
            if_alarm = alarm
            if_score = det.score(X[te].reset_index(drop=True))

    print("Prediction models ...", flush=True)
    fit_mask = (split != "test") & warm & (df.fault_type == "normal").to_numpy()
    normal_full = np.zeros(len(df), bool)
    normal_full[np.flatnonzero(te)] = mask & (test_df.fault_type == "normal").to_numpy()
    metrics["prediction"] = {}
    for target in PRED_TARGETS:
        res, fitted, pcols = fit_predictors(df, fit_mask, te & warm, normal_full, target)
        metrics["prediction"][target] = res
        for name, m in fitted.items():
            joblib.dump(dict(model=m, features=pcols, target=target, kind="residual correction"),
                        os.path.join(a.out, f"predict_{target}_{name}.joblib"))

    print("Latency benchmark ...", flush=True)
    metrics["latency"] = latency_benchmark(test_df, detectors["isolation_forest"])
    joblib.dump(detectors["isolation_forest"], os.path.join(a.out, "anomaly_isolation_forest.joblib"))

    for run in sorted(test_df.run_id.unique()):
        plot_run(test_df, if_score, detectors["isolation_forest"].threshold, if_alarm,
                 test_events, run, os.path.join(a.out, f"run{run}_twin_vs_measured.png"))

    pe = per_event_tables["isolation_forest"].rename(columns={"alarm_already_on": "already_on_if"}).merge(
        per_event_tables["lof"][["event_id", "detected", "delay_s"]], on="event_id", suffixes=("_if", "_lof")
    ).merge(per_event_tables["zscore"][["event_id", "detected", "delay_s"]].rename(
        columns={"detected": "detected_z", "delay_s": "delay_s_z"}), on="event_id")
    pe.to_csv(os.path.join(a.out, "per_event_detection.csv"), index=False)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    write_report(metrics, pe, os.path.join(a.out, "report.md"))
    print(f"Done in {time.time() - t0:.0f} s. Results in {a.out}/")
    print(json.dumps({k: v for k, v in metrics.items() if k != "prediction"}, indent=1))
    print(json.dumps(metrics["prediction"], indent=1))


def write_report(m, pe, path):
    L = ["# AquaTwin ML results (simulated data)", "",
         f"Data: {m['data']['hours']} h at 1 Hz ({m['data']['train_hours']} h train, "
         f"{m['data']['calibration_hours']} h calibration, {m['data']['test_hours']} h test), "
         f"{m['data']['fault_events']} injected faults, {m['data']['missing_readings_filled']} dropped readings filled.", "",
         "## Anomaly detection (test runs)", "",
         "| Detector | Events detected | Median delay (s) | Sample sensitivity | Sample specificity | False alarms / h |",
         "|---|---|---|---|---|---|"]
    names = {"isolation_forest": "Isolation Forest (primary)", "lof": "Local Outlier Factor",
             "zscore": "Residual z-score chart"}
    for k, n in names.items():
        s, e = m[k]["sample_level"], m[k]["event_level"]
        L.append(f"| {n} | {e['detection_rate']:.0%} | {e['median_delay_s']} | {s['sensitivity']:.1%} | "
                 f"{s['specificity']:.1%} | {e['false_alarms_per_hour']} |")
    L += ["", "## Per-event detection", "",
          "| Event | Fault | IF | IF delay (s) | LOF | LOF delay (s) | z-score | z delay (s) |",
          "|---|---|---|---|---|---|---|---|"]
    yn = lambda b: "yes" if b else "**no**"
    fmt = lambda d: "" if pd.isna(d) else str(int(d))
    for _, r in pe.iterrows():
        dif = fmt(r.delay_s_if) + (" (alarm already on)" if r.already_on_if else "")
        L.append(f"| {r.event_id} | {r.fault_type} | {yn(r.detected_if)} | {dif} | "
                 f"{yn(r.detected_lof)} | {fmt(r.delay_s_lof)} | {yn(r.detected_z)} | {fmt(r.delay_s_z)} |")
    L += ["", "## Prediction: hybrid twin (physics + ML residual correction), test runs", "",
          "Accuracy = 100 - MAPE. 'Normal' excludes fault windows; 'All test' includes them, where large errors are expected and are what the anomaly detector looks for.", ""]
    for tgt, res in m["prediction"].items():
        L += [f"**{tgt}**", "", "| Model | Normal: R² | Normal: accuracy (%) | All test: R² | All test: accuracy (%) |",
              "|---|---|---|---|---|"]
        for n in res["normal operation"]:
            a_, b_ = res["normal operation"][n], res["all test samples"][n]
            L.append(f"| {n} | {a_['r2']} | {a_['accuracy_pct']} | {b_['r2']} | {b_['accuracy_pct']} |")
        L.append("")
    lt = m["latency"]
    L += ["## Latency per 1 Hz sample", "",
          f"Features + Isolation Forest score: median {lt['features_plus_score_ms_median']} ms, "
          f"p95 {lt['features_plus_score_ms_p95']} ms. Uncached EPANET solve: {lt['epanet_uncached_solve_ms']} ms. "
          "Budget is 8 s end to end, so the remaining latency is the database and Grafana refresh.", ""]
    open(path, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
