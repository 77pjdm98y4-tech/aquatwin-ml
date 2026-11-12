"""
Export the simulated test runs in a format the Grafana dashboard reads.

    python export_dashboard_data.py            # writes dashboard/aquatwin_dashboard_data.csv

For every test run it writes, at 5-second resolution:
  * each sensor's measured value and the digital twin's expected value
  * the anomaly scores of the residual z-score chart and the Isolation Forest,
    divided by their alarm threshold (1.0 = threshold)
  * the alarm (z-score chart with the 15-of-20-seconds persistence rule)
  * the signal whose residual deviates most while the alarm is on
  * the injected fault (ground truth), for comparison

The four 3-hour test runs are placed back to back starting at START_UTC so the
dashboard shows one continuous 12-hour window. Grafana reads the file straight
from GitHub with the Infinity data source (see dashboard/README.md).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import pipeline as P

START_UTC = pd.Timestamp("2026-11-17 14:00:00", tz="UTC")  # 8:00 a.m. Central
RESAMPLE_S = 5

FRIENDLY = {
    "flow_Lpm": "Flow", "press_kPa": "Filter inlet pressure", "dp_filter_kPa": "Filter head loss",
    "level_cm": "Tank level", "turb_in_NTU": "Turbidity in", "turb_out_NTU": "Turbidity out",
    "do_mgL": "Dissolved oxygen", "ph": "pH", "cond_uScm": "Conductivity",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="dashboard/aquatwin_dashboard_data.csv")
    a = ap.parse_args()

    df = pd.read_csv(os.path.join(a.data, "sensor_data.csv"))
    df = P.clean(df)
    print("Digital twin ...", flush=True)
    df = P.add_twin(df, cache_path=os.path.join(a.data, "twin_predictions.csv"))
    print("Features and detectors ...", flush=True)
    X = P.build_features(df)
    feats = list(X.columns)
    split = df["split"].to_numpy()
    warm = df["t_s"].to_numpy() >= P.WARMUP_S
    tr, cal, te = (split == "train") & warm, (split == "calibration") & warm, split == "test"

    z_feats = [f for f in feats if f.endswith("_m120") or f.endswith("_slope300")]
    smooth = [f for f in feats if f.endswith(("_m30", "_m120", "_slope300", "_m60"))]
    zdet = P.Detector("zscore", z_feats).fit(X[tr]).calibrate(X[cal])
    ifdet = P.Detector("isolation_forest", smooth).fit(X[tr]).calibrate(X[cal])

    t = df[te].reset_index(drop=True)
    Xt = X[te].reset_index(drop=True)
    alarm = zdet.alarms(Xt, t.run_id.to_numpy())

    # which residual is furthest from normal (only meaningful while alarmed)
    Z = np.abs(zdet.scaler.transform(Xt[z_feats]))
    top = np.array(z_feats)[Z.argmax(axis=1)]
    base = pd.Series(top).str.replace(r"^r_|_m120$|_slope300$", "", regex=True)
    kind = np.where(pd.Series(top).str.endswith("_slope300"), " (drifting)", "")
    suspect = np.where(alarm, base.map(FRIENDLY).fillna(base) + kind, "No alarm")

    runs = sorted(t.run_id.unique())
    order = {r: i for i, r in enumerate(runs)}
    secs = t.run_id.map(order) * (t.t_s.max() + 1) + t.t_s
    out = pd.DataFrame({"time": START_UTC + pd.to_timedelta(secs, unit="s")})
    out["run"] = t.run_id
    for c in P.ALL_SENSORS:
        out[c] = t[c]
    for c in P.SENSORS:
        out[f"twin_{c}"] = t[f"twin_{c}"]
    out["score_zscore"] = zdet.score(Xt) / zdet.threshold
    out["score_iforest"] = ifdet.score(Xt) / ifdet.threshold
    out["alarm"] = alarm.astype(int)
    out["suspect_signal"] = suspect
    out["injected_fault"] = t.fault_type.str.replace("_", " ")
    out.loc[t.t_s < P.WARMUP_S, ["score_zscore", "score_iforest"]] = np.nan  # rolling windows still filling

    # 5-second bins: mean of readings, max of scores/alarm, most severe label
    out["bin"] = (out.time - START_UTC).dt.total_seconds() // RESAMPLE_S
    num = [c for c in out.columns if c not in ("time", "bin", "suspect_signal", "injected_fault", "run")]
    agg = {c: "mean" for c in num}
    agg.update(score_zscore="max", score_iforest="max", alarm="max", run="first",
               suspect_signal=lambda s: next((v for v in s if v != "No alarm"), "No alarm"),
               injected_fault=lambda s: next((v for v in s if v != "normal"), "normal"))
    g = out.groupby("bin").agg(agg)
    g.insert(0, "time", START_UTC + pd.to_timedelta(g.index * RESAMPLE_S, unit="s"))
    g.insert(1, "time_epoch_ms", (g.time.astype("int64") // 10**6).astype("int64"))
    g["time"] = g.time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    g["alarm"] = g.alarm.astype(int)
    roundto = {c: 3 for c in num}
    roundto.update(ph=3, score_zscore=3, score_iforest=3, do_mgL=3, twin_do_mgL=3, twin_ph=3)
    g = g.round(roundto)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    g.to_csv(a.out, index=False)
    print(f"Wrote {len(g):,} rows ({os.path.getsize(a.out) / 1e6:.1f} MB) to {a.out}")
    print(f"Time window: {g.time.iloc[0]} to {g.time.iloc[-1]}")


if __name__ == "__main__":
    main()
