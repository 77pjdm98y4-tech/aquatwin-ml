# AquaTwin ML results (simulated data)

Data: 30.0 h at 1 Hz (11.7 h train, 5.8 h calibration, 12.0 h test), 12 injected faults, 1098 dropped readings filled.

## Anomaly detection (test runs)

| Detector | Events detected | Median delay (s) | Sample sensitivity | Sample specificity | False alarms / h |
|---|---|---|---|---|---|
| Isolation Forest (primary) | 58% | 112.0 | 35.7% | 96.9% | 0.73 |
| Local Outlier Factor | 100% | 44.0 | 95.0% | 78.1% | 2.0 |
| Residual z-score chart | 100% | 69.5 | 94.1% | 77.5% | 0.36 |

## Per-event detection

| Event | Fault | IF | IF delay (s) | LOF | LOF delay (s) | z-score | z delay (s) |
|---|---|---|---|---|---|---|---|
| 0 | leak | **no** |  | yes | 38 | yes | 93 |
| 1 | partial_blockage | yes | 112 | yes | 50 | yes | 71 |
| 2 | turbidity_spike | yes | 41 | yes | 17 | yes | 25 |
| 3 | low_do | **no** |  | yes | 70 | yes | 68 |
| 4 | pump_slowdown | yes | 139 | yes | 11 | yes | 26 |
| 5 | pump_slowdown | yes | 49 | yes | 16 | yes | 0 |
| 6 | sensor_drift | **no** |  | yes | 78 | yes | 78 |
| 7 | leak | yes | 119 | yes | 95 | yes | 105 |
| 8 | sensor_drift | **no** |  | yes | 102 | yes | 103 |
| 9 | turbidity_spike | yes | 656 | yes | 20 | yes | 23 |
| 10 | low_do | **no** |  | yes | 70 | yes | 77 |
| 11 | partial_blockage | yes | 39 | yes | 0 | yes | 0 |

## Prediction: hybrid twin (physics + ML residual correction), test runs

Accuracy = 100 - MAPE. 'Normal' excludes fault windows; 'All test' includes them, where large errors are expected and are what the anomaly detector looks for.

**turb_out_NTU**

| Model | Normal: R² | Normal: accuracy (%) | All test: R² | All test: accuracy (%) |
|---|---|---|---|---|
| physics twin only | 0.757 | 74.7 | 0.648 | 71.5 |
| hybrid: physics + random_forest | 0.966 | 92.7 | 0.709 | 88.2 |
| hybrid: physics + xgboost | 0.963 | 92.4 | 0.711 | 88.0 |

**press_kPa**

| Model | Normal: R² | Normal: accuracy (%) | All test: R² | All test: accuracy (%) |
|---|---|---|---|---|
| physics twin only | 0.96 | 97.1 | 0.773 | 95.5 |
| hybrid: physics + random_forest | 0.99 | 98.6 | 0.804 | 96.7 |
| hybrid: physics + xgboost | 0.99 | 98.6 | 0.803 | 96.7 |

**dp_filter_kPa**

| Model | Normal: R² | Normal: accuracy (%) | All test: R² | All test: accuracy (%) |
|---|---|---|---|---|
| physics twin only | 0.931 | 93.3 | 0.729 | 91.1 |
| hybrid: physics + random_forest | 0.99 | 98.0 | 0.785 | 95.2 |
| hybrid: physics + xgboost | 0.99 | 98.0 | 0.784 | 95.2 |

## Latency per 1 Hz sample

Features + Isolation Forest score: median 31.49 ms, p95 51.23 ms. Uncached EPANET solve: 14.9 ms. Budget is 8 s end to end, so the remaining latency is the database and Grafana refresh.
