"""
Build dashboard/aquatwin_dashboard.json, an importable Grafana dashboard that reads
dashboard/aquatwin_dashboard_data.csv from GitHub through the Infinity data source.

    python dashboard/build_dashboard.py
    python dashboard/build_dashboard.py --url https://.../other.csv   # different data file
"""
from __future__ import annotations

import argparse
import json
import os

DEFAULT_URL = ("https://raw.githubusercontent.com/77pjdm98y4-tech/aquatwin-ml/"
               "main/dashboard/aquatwin_dashboard_data.csv")
DS = {"type": "yesoreyeram-infinity-datasource", "uid": "${DS_INFINITY}"}
TIME_FROM, TIME_TO = "2026-11-17T14:00:00.000Z", "2026-11-18T02:00:00.000Z"

MEASURED = "#3987e5"   # blue, measured reading
TWIN = "#c3c2b7"       # light gray dashed, digital twin expectation
GOOD, CRITICAL, WARN = "#0ca30c", "#d03b3b", "#fab219"

_ids = iter(range(1, 1000))


def target(url, cols, ref="A"):
    """Infinity query: CSV from URL, backend parser, explicit column types."""
    columns = [{"selector": "time_epoch_ms", "text": "Time", "type": "timestamp_epoch"}]
    columns += [{"selector": s, "text": t, "type": ty} for s, t, ty in cols]
    return {"refId": ref, "datasource": DS, "type": "csv", "source": "url", "format": "table",
            "parser": "backend", "url": url, "url_options": {"method": "GET", "data": ""},
            "root_selector": "", "columns": columns, "filters": []}


def color_override(name, color, dashed=False, width=1):
    props = [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}},
             {"id": "custom.lineWidth", "value": width}]
    if dashed:
        props.append({"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [8, 6]}})
    return {"matcher": {"id": "byName", "options": name}, "properties": props}


def timeseries(title, url, cols, grid, unit="none", overrides=(), custom=None, thresholds=None,
               description=""):
    c = {"drawStyle": "line", "lineInterpolation": "linear", "lineWidth": 1, "fillOpacity": 0,
         "showPoints": "never", "spanNulls": False, "axisBorderShow": False,
         "thresholdsStyle": {"mode": "off"}}
    c.update(custom or {})
    return {"id": next(_ids), "type": "timeseries", "title": title, "description": description,
            "datasource": DS, "gridPos": grid, "targets": [target(url, cols)],
            "fieldConfig": {"defaults": {"unit": unit, "custom": c, "color": {"mode": "palette-classic"},
                                         "thresholds": thresholds or {"mode": "absolute", "steps": [
                                             {"color": "green", "value": None}]}},
                            "overrides": list(overrides)},
            "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                        "tooltip": {"mode": "multi", "sort": "none"}}}


def sensor_panel(title, url, col, unit, grid, twin=True):
    cols = [(col, "Measured", "number")]
    ov = [color_override("Measured", MEASURED, width=1)]
    if twin:
        cols.append((f"twin_{col}", "Twin expected", "number"))
        ov.append(color_override("Twin expected", TWIN, dashed=True, width=2))
    return timeseries(title, url, cols, grid, unit=unit, overrides=ov,
                      description="Blue: sensor reading. Dashed gray: what the digital twin expects "
                                  "from the current pump, valve and aeration settings. A gap "
                                  "between them is what the anomaly detector looks for." if twin else
                                  "Measured water temperature (used by the twin for DO saturation).")


def stat(title, url, cols, grid, calc, unit="none", mappings=(), thresholds=None, field=None,
         color_mode="value", description=""):
    return {"id": next(_ids), "type": "stat", "title": title, "description": description,
            "datasource": DS, "gridPos": grid, "targets": [target(url, cols)],
            "fieldConfig": {"defaults": {"unit": unit, "mappings": list(mappings),
                                         "color": {"mode": "thresholds"},
                                         "thresholds": thresholds or {"mode": "absolute", "steps": [
                                             {"color": "text", "value": None}]}},
                            "overrides": []},
            "options": {"reduceOptions": {"values": False, "calcs": [calc],
                                          "fields": f"/^{field}$/" if field else ""},
                        "colorMode": color_mode, "graphMode": "none", "justifyMode": "center",
                        "textMode": "value", "orientation": "auto", "wideLayout": True,
                        "showPercentChange": False}}


def row(title, y):
    return {"id": next(_ids), "type": "row", "title": title, "collapsed": False, "panels": [],
            "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}}


def build(url):
    alarm_map = [{"type": "value", "options": {
        "0": {"text": "Normal", "color": GOOD, "index": 0},
        "1": {"text": "ALARM", "color": CRITICAL, "index": 1}}}]
    fault_map = [{"type": "value", "options": {
        "normal": {"text": "No fault", "color": "transparent", "index": 0},
        "turbidity spike": {"color": WARN, "index": 1},
        "pump slowdown": {"color": WARN, "index": 2},
        "partial blockage": {"color": WARN, "index": 3},
        "leak": {"color": WARN, "index": 4},
        "low do": {"text": "low DO", "color": WARN, "index": 5},
        "sensor drift": {"color": WARN, "index": 6}}}]
    ratio_thr = {"mode": "absolute", "steps": [{"color": GOOD, "value": None},
                                               {"color": CRITICAL, "value": 1}]}
    P = []
    P.append({"id": next(_ids), "type": "text", "title": "", "gridPos": {"h": 3, "w": 24, "x": 0, "y": 0},
              "options": {"mode": "markdown", "content":
                  "**Simulated test data** (4 runs × 3 h, 12 injected faults), replayed from the AquaTwin "
                  "pipeline on GitHub. Blue lines are sensor readings; dashed gray lines are the digital "
                  "twin's expected values. The alarm uses the residual z-score chart (alarm when the score "
                  "stays above 1 for 15 of 20 seconds). After a turbidity slug or an aeration failure the "
                  "alarm can stay on 30–60 min, because the water stays abnormal until the filter and "
                  "aeration bring it back to what the twin expects."}})
    # status row
    P.append(stat("Detector status (latest)", url, [("alarm", "Alarm", "number")],
                  {"h": 5, "w": 5, "x": 0, "y": 3}, "lastNotNull", mappings=alarm_map,
                  thresholds={"mode": "absolute", "steps": [{"color": GOOD, "value": None},
                                                            {"color": CRITICAL, "value": 1}]},
                  color_mode="background", field="Alarm"))
    P.append(stat("Suspected signal (latest)", url, [("suspect_signal", "Suspected signal", "string")],
                  {"h": 5, "w": 5, "x": 5, "y": 3}, "lastNotNull", field="Suspected signal",
                  description="Signal whose residual is furthest from normal while the alarm is on."))
    P.append(stat("Injected fault (ground truth)", url, [("injected_fault", "Injected fault", "string")],
                  {"h": 5, "w": 5, "x": 10, "y": 3}, "lastNotNull", mappings=fault_map,
                  field="Injected fault",
                  description="What the simulator actually injected. Not available on real data; "
                              "shown here to check the detector."))
    P.append(stat("Peak anomaly score", url, [("score_zscore", "Score", "number")],
                  {"h": 5, "w": 4, "x": 15, "y": 3}, "max", unit="x", thresholds=ratio_thr,
                  field="Score", description="Residual z-score chart, as a multiple of the alarm threshold."))
    P.append(stat("Time in alarm", url, [("alarm", "Alarm", "number")],
                  {"h": 5, "w": 5, "x": 19, "y": 3}, "mean", unit="percentunit", field="Alarm",
                  description="Share of the 12-hour window with the alarm on."))
    # detection
    P.append(row("Anomaly detection", 8))
    tl = {"id": next(_ids), "type": "state-timeline", "title": "Alarm vs injected fault",
          "datasource": DS, "gridPos": {"h": 5, "w": 24, "x": 0, "y": 9},
          "targets": [target(url, [("alarm", "Detector alarm", "number"),
                                   ("injected_fault", "Injected fault", "string")])],
          "fieldConfig": {"defaults": {"mappings": alarm_map + fault_map,
                                       "color": {"mode": "fixed", "fixedColor": "transparent"},
                                       "custom": {"fillOpacity": 85, "lineWidth": 0}},
                          "overrides": []},
          "options": {"showValue": "auto", "mergeValues": True, "rowHeight": 0.8, "alignValue": "center",
                      "legend": {"showLegend": False}, "tooltip": {"mode": "single"}}}
    P.append(tl)
    P.append(timeseries(
        "Anomaly score (× alarm threshold)", url,
        [("score_zscore", "Residual z-score chart", "number"), ("score_iforest", "Isolation Forest", "number")],
        {"h": 8, "w": 24, "x": 0, "y": 14}, unit="x",
        overrides=[color_override("Residual z-score chart", MEASURED, width=1),
                   color_override("Isolation Forest", "#e87ba4", width=1)],
        custom={"scaleDistribution": {"type": "log", "log": 10}, "thresholdsStyle": {"mode": "dashed"}},
        thresholds=ratio_thr,
        description="1.0 (dashed red line) is each detector's alarm threshold, set so a normal "
                    "calibration run stays below it 99.5% of the time. Log scale."))
    # hydraulics
    P.append(row("Hydraulics", 22))
    for i, (t, c, u) in enumerate([("Flow", "flow_Lpm", "flowlpm"),
                                   ("Filter inlet pressure", "press_kPa", "pressurekpa"),
                                   ("Filter head loss", "dp_filter_kPa", "pressurekpa"),
                                   ("Tank level", "level_cm", "suffix: cm")]):
        P.append(sensor_panel(t, url, c, u, {"h": 8, "w": 6, "x": 6 * i, "y": 23}))
    # water quality
    P.append(row("Water quality", 31))
    for i, (t, c, u) in enumerate([("Turbidity, filter inlet", "turb_in_NTU", "suffix: NTU"),
                                   ("Turbidity, filter outlet", "turb_out_NTU", "suffix: NTU"),
                                   ("Dissolved oxygen", "do_mgL", "suffix: mg/L")]):
        P.append(sensor_panel(t, url, c, u, {"h": 8, "w": 8, "x": 8 * i, "y": 32}))
    for i, (t, c, u, tw) in enumerate([("pH", "ph", "none", True),
                                       ("Conductivity", "cond_uScm", "suffix: µS/cm", True),
                                       ("Temperature", "temp_C", "celsius", False)]):
        P.append(sensor_panel(t, url, c, u, {"h": 8, "w": 8, "x": 8 * i, "y": 40}, twin=tw))

    return {
        "__inputs": [{"name": "DS_INFINITY", "label": "Infinity", "description": "",
                      "type": "datasource", "pluginId": "yesoreyeram-infinity-datasource",
                      "pluginName": "Infinity"}],
        "__requires": [{"type": "datasource", "id": "yesoreyeram-infinity-datasource", "name": "Infinity",
                        "version": "2.0.0"},
                       {"type": "grafana", "id": "grafana", "name": "Grafana", "version": "10.0.0"}],
        "uid": "aquatwin-sim", "title": "AquaTwin Digital Twin (simulated data)",
        "tags": ["aquatwin", "digital-twin", "simulated"], "timezone": "browser",
        "editable": True, "graphTooltip": 1, "schemaVersion": 39, "version": 1,
        "time": {"from": TIME_FROM, "to": TIME_TO}, "refresh": "",
        "templating": {"list": []}, "annotations": {"list": []}, "links": [],
        "panels": P,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "aquatwin_dashboard.json"))
    a = ap.parse_args()
    with open(a.out, "w") as f:
        json.dump(build(a.url), f, indent=2, ensure_ascii=False)
    print("Wrote", a.out)
