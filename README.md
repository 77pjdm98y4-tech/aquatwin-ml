# AquaTwin ML starter kit

Simulated bench-top data, a physics-based digital twin, anomaly detection and prediction models, and an evaluation harness that reports the Interim Report metrics. Built so Track B can run now and switch to real bench-top data in mid-November by replacing one CSV.

## Files

| File | What it does |
|---|---|
| `physics.py` | Testbed parameters, EPANET/WNTR hydraulic model, Carman–Kozeny filter, DO transfer model, 1 Hz time-stepping engine. Shared by the simulator and the twin. |
| `simulate.py` | Generates labeled 1 Hz data with the six faults from Section 3.2. Writes `data/sensor_data.csv` and `data/events.csv`. |
| `pipeline.py` | Data cleaning → digital twin → residual features → anomaly detectors → hybrid prediction models → metrics, figures, report. |
| `AquaTwin_ML_Colab.ipynb` | Runs everything in Google Colab. |
| `results/` | `report.md`, `metrics.json`, per-event table, one figure per test run. |

## Run it

**Colab:** open `AquaTwin_ML_Colab.ipynb` in Colab and choose Runtime → Run all. The first cell asks you to pick `aquatwin_ml.zip` if it isn't already uploaded (about 6 minutes total).

**Locally:**
```
pip install -r requirements.txt
python simulate.py          # 30 h of data, ~30 s
python pipeline.py          # ~5 min, writes results/
```
Options: `python simulate.py --runs 12 --hours 2 --seed 3`.

## What the simulated data contains

- **10 runs × 3 h at 1 Hz** (30 h). Runs 0–3: normal, used for training. Runs 4–5: normal, used only to set the alarm threshold. Runs 6–9: normal operation with 3 injected faults each, used for testing.
- **Operating changes** every 20–40 min: pump speed (0.70 / 0.85 / 1.00), ball valve (100 / 70 / 50 % open), aeration level (0–3), plus 1–2 planned, logged kaolin doses per run.
- **Faults**, each twice: turbidity spike (unlogged sediment slug), pump slowdown (speed 12–25 % below command), partial blockage (filter resistance ×1.8–3, backwash at the end), leak (0.3–0.5 L/min after the pump, refill at the end), low DO (air pump fails while commanded on), sensor drift (pH, DO, conductivity or outlet turbidity).
- **Sensor layer:** Gaussian noise sized to typical low-cost probes, per-run calibration offsets, quantization, air-bubble spikes on the turbidity probes, and 0.1 % random dropouts.

### CSV schema (match this when logging the real testbed)

`timestamp, run_id, t_s, pump_speed_cmd, valve_open_cmd, aeration_cmd, dose_NTU, backwash, refill, flow_Lpm, press_kPa, dp_filter_kPa, level_cm, turb_in_NTU, turb_out_NTU, do_mgL, temp_C, ph, cond_uScm, fault_type, fault_id, split`

The command and event columns (`dose_NTU`, `backwash`, `refill`) are the event log from the experimental protocol. The twin needs them, so the protocol should record every manual change with a timestamp.

## How the digital twin works

The twin runs the same physics engine as the simulator, but with no faults, no noise, and parameters a few percent off (pump head −2 %, grain size +3 %, KLa −8 %, removal efficiency −3 %), the way a real uncalibrated model would be. It takes the logged commands plus measured temperature, and it estimates the filter's clogging state from measured head loss with a 30-minute time constant. That last step is simple data assimilation: slow, normal clogging is absorbed into the model, while a sudden blockage still shows up as a residual.

Every feature the detectors see is a **residual** (measured − twin), smoothed over trailing 30 s and 120 s windows, plus 5-minute slopes for drift. Trailing windows only, so everything is computable in real time.

## Results on simulated data

Full tables are in `results/report.md`.

**Prediction (hybrid twin = physics + Random Forest residual correction), normal operation in test runs**

| Target | Physics only | Hybrid (physics + RF) |
|---|---|---|
| Outlet turbidity | R² 0.76, 74.7 % accuracy | R² 0.97, 92.7 % accuracy |
| Filter-inlet pressure | R² 0.96, 97.1 % | R² 0.99, 98.6 % |
| Filter head loss | R² 0.93, 93.3 % | R² 0.99, 98.0 % |

Accuracy here is 100 − MAPE. Both turbidity and pressure clear the 85 % target, and head loss is within the < 10 % steady-state error target. XGBoost scores the same as Random Forest.

**Anomaly detection, test runs (12 fault events)**

| Detector | Events detected | Median delay | False alarms / h |
|---|---|---|---|
| Isolation Forest (planned primary) | 7 of 12 | 112 s | 0.7 |
| Local Outlier Factor | 12 of 12 | 44 s | 2.0 |
| Residual z-score control chart | 12 of 12 | 70 s | 0.4 |

**Latency:** feature computation plus scoring takes about 31 ms per sample (p95 51 ms); an uncached EPANET solve takes about 15 ms. The 8-second budget will be spent in the database and Grafana refresh, not the models.

## Findings worth discussing with the team

1. **Isolation Forest misses single-sensor faults.** It missed both sensor drifts and both low-DO events even though they are obvious in the plots (see `results/run8_twin_vs_measured.png`). With 30-plus residual features, a fault that moves only one or two of them barely shortens the isolation path. The per-residual z-score chart catches all 12 with the fewest false alarms. Suggestion: make the residual control chart the primary detector, keep Isolation Forest for multi-sensor patterns, and report both. This changes Appendix B, so it's a team decision.
2. **Forecasting ahead isn't the right prediction task.** I first trained models to forecast turbidity and pressure 60 s ahead. They lost to "value stays the same" because the signal is dominated by operator changes no model can foresee. The hybrid twin, which predicts what each sensor *should* read given the current settings, is the useful framing. It also works as a virtual sensor if a probe fails. Worth wording the 85 % objective that way in the final report.
3. **The twin must track slow states.** Before adding clogging assimilation, every turbidity spike left the filter dirtier than the twin assumed, and alarms stayed on for 40 minutes afterward. Any slow state the twin can't see (clogging, probe fouling) needs to be estimated from measurements.
4. **Alarm tails.** Sample-level specificity is only 75–78 % for LOF and the z-score chart, mostly because alarms stay on for a while after a fault ends. A reset rule or a faster recovery model would raise it.

## When real data arrives

1. Replace every `TODO` in `TestbedParams` (`physics.py`) with the as-built dimensions: tank footprint and depth, pipe diameter and lengths, filter column diameter, bed depth, sand size, and the pump curve from its datasheet.
2. Fit `kla_per_h` for each aeration level from a reaeration test: strip DO with sodium sulfite or nitrogen, turn on aeration, and fit ln(Cs − C) versus time. The slope is −KLa.
3. Fit `removal_eff` from inlet and outlet turbidity at steady state.
4. Log data in the CSV schema above, mark each run `train`, `calibration` or `test`, and run `python pipeline.py --data <folder>`.
5. Report simulated and physical results separately, as the interim report promises.

## Known limitations

- The supply and return tanks are modeled as one mixed volume.
- Carman–Kozeny assumes laminar flow through the bed, but the bed Reynolds number is about 12 at full pump speed with these placeholder dimensions. If the real filter velocity is this high, switch to the Ergun equation.
- pH and conductivity use simple empirical models, not chemistry.
- Results on simulated data are optimistic: the simulator and twin share their model structure, so only parameter error separates them. Expect lower scores on real data, which is exactly what the validation phase should measure.
