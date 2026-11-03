"""
AquaTwin physics core.

One engine, two uses:
  * simulate.py runs it WITH faults, sensor noise and the "true" parameters
    to stand in for the bench-top testbed until real data exists.
  * pipeline.py runs it WITHOUT faults or noise, with slightly-off
    "model" parameters, as the digital twin's physics prediction. The ML
    models then learn from the residual (measured - twin).

Models (as in Interim Report, Section 3.5):
  Hydraulics      EPA EPANET 2.2 via WNTR (pump curve, pipes, valve, filter)
  Filter head loss Carman-Kozeny, entered into EPANET as a linear
                  GPV head-loss curve, scaled by a clogging multiplier
  Turbidity       completely mixed tank + filter removal efficiency
  Dissolved O2    dC/dt = KLa (Cs(T) - C) - oxygen demand
  pH, conductivity simple first-order empirical models (placeholders)

Every number marked TODO should be replaced with the real bench-top value
once the build is final (pipe lengths, tank size, filter column, pump).
"""
from __future__ import annotations

import functools
import os
import tempfile
from dataclasses import dataclass, replace

import numpy as np
import wntr

G = 9.81          # m/s2
RHO = 998.0       # kg/m3, water at ~21 C
MU = 1.0e-3       # Pa.s
LPM = 1 / 60000   # 1 L/min in m3/s
NTU_TO_MGL = 1.5  # approx TSS (mg/L) per NTU for kaolin suspensions


@dataclass(frozen=True)
class TestbedParams:
    # ---- Tank (supply + return treated as one recirculating volume) ----
    tank_area_m2: float = 0.25          # TODO ~50 x 50 cm footprint
    level_nominal_m: float = 0.30       # TODO normal water depth
    # ---- Elevations ----
    filter_elev_m: float = 0.50         # TODO filter inlet/outlet height
    discharge_head_m: float = 1.00      # TODO free discharge back into tank
    # ---- Piping (PVC) ----
    pipe_d_m: float = 0.019             # TODO 3/4 in nominal
    L_suction_m: float = 1.0            # TODO
    L_pump_valve_m: float = 1.5         # TODO
    L_filter_return_m: float = 2.0      # TODO
    hw_C: float = 150.0                 # Hazen-Williams C for PVC
    # ---- Pump (single-point curve, EPANET fits H = A - B Q^2) ----
    pump_design_flow_Lpm: float = 8.0   # TODO from pump datasheet
    pump_design_head_m: float = 3.5     # TODO from pump datasheet
    # ---- Ball valve: K = K_open + K_throttle * ((1 - opening) / opening)^2 ----
    valve_K_open: float = 0.1
    valve_K_throttle: float = 30.0
    # ---- Filter column (Carman-Kozeny) ----
    filter_diam_m: float = 0.15         # TODO 6 in column
    bed_depth_m: float = 0.30           # TODO
    grain_d_m: float = 0.6e-3           # TODO effective sand size
    porosity: float = 0.42
    k_clog_per_g: float = 0.03          # head-loss increase per g deposited
    removal_eff: float = 0.80           # fraction of turbidity removed per pass
    # ---- Turbidity ----
    turb_baseline_NTU: float = 5.0      # tank turbidity at nominal steady state
    # ---- Dissolved oxygen ----
    kla_per_h: tuple = (0.0, 4.0, 8.0, 12.0)  # aeration level 0..3 (TODO fit)
    kla_splash_per_h: float = 1.0       # re-aeration at the return, at nominal flow
    o2_demand_mgL_h: float = 3.0        # oxygen uptake (organics/biofilm)
    # ---- pH / conductivity (empirical placeholders) ----
    ph_base: float = 7.20
    ph_aeration_gain: float = 0.25      # CO2 stripping raises pH at max aeration
    ph_tau_s: float = 900.0
    cond_base_uScm: float = 450.0
    cond_per_NTU: float = 0.6
    # ---- Temperature ----
    temp_base_C: float = 22.5

    @property
    def tank_volume_m3(self) -> float:
        return self.tank_area_m2 * self.level_nominal_m

    @property
    def filter_area_m2(self) -> float:
        return np.pi * self.filter_diam_m ** 2 / 4

    def carman_kozeny_coef(self) -> float:
        """Clean-bed head loss per unit flow, h_L = coef * Q  [m per m3/s]."""
        e, d = self.porosity, self.grain_d_m
        per_velocity = 180 * MU * (1 - e) ** 2 * self.bed_depth_m / (RHO * G * e ** 3 * d ** 2)
        return per_velocity / self.filter_area_m2

    def bed_reynolds(self, q_m3s: float) -> float:
        v = q_m3s / self.filter_area_m2
        return RHO * v * self.grain_d_m / (MU * (1 - self.porosity))


def twin_params(p: TestbedParams) -> TestbedParams:
    """Model parameters the twin uses: deliberately a few percent off the
    'true' simulator values, the way a real model is never perfectly calibrated."""
    return replace(
        p,
        pump_design_head_m=p.pump_design_head_m * 0.98,
        grain_d_m=p.grain_d_m * 1.03,
        removal_eff=p.removal_eff * 0.97,
        kla_per_h=tuple(k * 0.92 for k in p.kla_per_h),
        o2_demand_mgL_h=p.o2_demand_mgL_h * 1.05,
        turb_baseline_NTU=p.turb_baseline_NTU * 1.04,
    )


# --------------------------------------------------------------------------
# Hydraulics (EPANET through WNTR)
# --------------------------------------------------------------------------
_EPA_DIR = os.environ.get("AQUATWIN_EPA_DIR", os.path.join(tempfile.gettempdir(), "aquatwin_epa"))
os.makedirs(_EPA_DIR, exist_ok=True)


def build_network(p: TestbedParams, speed=1.0, valve_open=1.0, filter_mult=1.0,
                  leak_coef=0.0, level_m=None) -> wntr.network.WaterNetworkModel:
    """Steady-state EPANET model of the bench-top loop.

    TANK -> suction pipe -> PUMP -> pipe (leak node J_PD) -> ball VALVE
         -> J_FIN (pressure sensor) -> FILTER (GPV, Carman-Kozeny curve)
         -> J_FOUT -> return pipe (flow sensor) -> free discharge to tank
    """
    level = p.level_nominal_m if level_m is None else level_m
    wn = wntr.network.WaterNetworkModel()
    wn.options.time.duration = 0
    wn.add_reservoir("TANK", base_head=level)
    wn.add_junction("J_SUC", elevation=0.0)
    wn.add_junction("J_PD", elevation=0.0)
    wn.add_junction("J_V", elevation=p.filter_elev_m * 0.6)
    wn.add_junction("J_FIN", elevation=p.filter_elev_m)
    wn.add_junction("J_FOUT", elevation=p.filter_elev_m)
    wn.add_reservoir("RET", base_head=p.discharge_head_m)
    d, C = p.pipe_d_m, p.hw_C
    wn.add_pipe("P_SUC", "TANK", "J_SUC", length=p.L_suction_m, diameter=d, roughness=C)
    wn.add_curve("PUMP_CURVE", "HEAD", [(p.pump_design_flow_Lpm * LPM, p.pump_design_head_m)])
    wn.add_pump("PUMP", "J_SUC", "J_PD", pump_type="HEAD", pump_parameter="PUMP_CURVE",
                speed=max(speed, 0.05))
    wn.add_pipe("P_PV", "J_PD", "J_V", length=p.L_pump_valve_m, diameter=d, roughness=C)
    f = min(max(valve_open, 0.05), 1.0)
    K = p.valve_K_open + p.valve_K_throttle * ((1 - f) / f) ** 2
    wn.add_valve("VALVE", "J_V", "J_FIN", diameter=d, valve_type="TCV", initial_setting=K)
    q_max = 1e-3
    wn.add_curve("FILTER_CURVE", "HEADLOSS",
                 [(0.0, 0.0), (q_max, p.carman_kozeny_coef() * filter_mult * q_max)])
    wn.add_valve("FILTER", "J_FIN", "J_FOUT", diameter=p.filter_diam_m,
                 valve_type="GPV", initial_setting="FILTER_CURVE")
    wn.add_pipe("P_RET", "J_FOUT", "RET", length=p.L_filter_return_m, diameter=d, roughness=C)
    if leak_coef > 0:
        wn.get_node("J_PD").emitter_coefficient = leak_coef
    return wn


@functools.lru_cache(maxsize=50_000)
def _solve_cached(p, speed, valve_open, filter_mult, leak_coef, level_m):
    wn = build_network(p, speed, valve_open, filter_mult, leak_coef, level_m)
    res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(_EPA_DIR, f"aq{os.getpid()}"))
    q = res.link["flowrate"].iloc[0]
    pr = res.node["pressure"].iloc[0]
    hd = res.node["head"].iloc[0]
    q_pump = float(q["PUMP"])
    q_filter = float(q["FILTER"])
    return dict(
        q_pump_Lpm=q_pump / LPM,
        flow_Lpm=q_filter / LPM,
        leak_Lpm=max(q_pump - q_filter, 0.0) / LPM,
        press_kPa=float(pr["J_FIN"]) * RHO * G / 1000,
        dp_filter_kPa=float(hd["J_FIN"] - hd["J_FOUT"]) * RHO * G / 1000,
    )


def solve_hydraulics(p, speed, valve_open, filter_mult=1.0, leak_coef=0.0, level_m=None):
    """Rounded inputs so the cache gets reused across 1 Hz time steps."""
    level = p.level_nominal_m if level_m is None else level_m
    return _solve_cached(p, round(speed, 3), round(valve_open, 2), round(filter_mult, 2),
                         round(leak_coef, 8), round(level, 3))


# --------------------------------------------------------------------------
# Water quality helpers
# --------------------------------------------------------------------------
def do_saturation(T_C):
    """DO saturation (mg/L) in fresh water at 1 atm (APHA polynomial fit)."""
    T = np.asarray(T_C, dtype=float)
    return 14.652 - 0.41022 * T + 0.0079910 * T ** 2 - 0.000077774 * T ** 3


# --------------------------------------------------------------------------
# Time-stepping engine (1 s steps)
# --------------------------------------------------------------------------
def run_engine(schedule, p: TestbedParams, faults=None, measured_temp=None,
               measured_dp=None, measured_flow=None, assim_tau_s=1800.0, seed=0):
    """Integrate the testbed for every row of `schedule`.

    schedule: DataFrame at 1 Hz with columns
        run_id, t_s, pump_speed_cmd, valve_open_cmd, aeration_cmd,
        dose_NTU (planned, logged kaolin dose added this second),
        backwash (1 at the second the filter is backwashed), refill (1 when topped up)
    faults: DataFrame of fault-effect columns aligned with schedule (or None):
        f_speed_mult, f_filter_mult, f_leak_coef, f_kla_mult, f_dose_NTU
    measured_temp: if given (twin mode), use it for DO saturation and
        conductivity instead of the simulated temperature.
    measured_dp, measured_flow: if given (twin mode), the twin estimates the
        filter's clogging state from them (data assimilation) with a slow
        time constant `assim_tau_s`, instead of relying only on its own
        deposit balance. Slow, legitimate clogging is absorbed; sudden
        blockages still show up as residuals.
    Returns a dict of numpy arrays holding the TRUE process values.
    """
    n = len(schedule)
    rng = np.random.default_rng(seed)
    S = {c: schedule[c].to_numpy() for c in schedule.columns}
    F = None if faults is None else {c: faults[c].to_numpy() for c in faults.columns}

    out = {k: np.zeros(n) for k in [
        "flow_Lpm", "q_pump_Lpm", "leak_Lpm", "press_kPa", "dp_filter_kPa",
        "level_cm", "turb_in_NTU", "turb_out_NTU", "do_mgL", "temp_C", "ph", "cond_uScm",
        "filter_mult"]}

    V0 = p.tank_volume_m3
    hyd_nom = solve_hydraulics(p, 1.0, 1.0)
    q_nom = hyd_nom["flow_Lpm"] * LPM
    # background resuspension that holds the tank at turb_baseline at nominal flow
    r_bg = p.turb_baseline_NTU * q_nom * p.removal_eff / V0  # NTU/s

    run_prev = None
    tau_flow = 2.0  # s, pump/flow response lag
    for i in range(n):
        run = S["run_id"][i]
        if run != run_prev:  # new run: reset state (filter backwashed, tank topped up)
            V = V0
            M = 0.0
            Ct = p.turb_baseline_NTU
            T_lag = 0.0
            Cdo = None
            ph = p.ph_base
            flow_state = None
            mult_hat = 1.0
            t_offset = rng.uniform(0, 86400)
            run_prev = run
        t = S["t_s"][i]

        if S["backwash"][i]:
            M = 0.0
            mult_hat = 1.0
        if S["refill"][i]:
            V = V0

        speed = S["pump_speed_cmd"][i] * (F["f_speed_mult"][i] if F else 1.0)
        fmult = (1 + p.k_clog_per_g * M) * (F["f_filter_mult"][i] if F else 1.0)
        if measured_dp is not None and measured_flow[i] > 1.0:
            clean_dp_kPa = p.carman_kozeny_coef() * measured_flow[i] * LPM * RHO * G / 1000
            mult_hat += (measured_dp[i] / clean_dp_kPa - mult_hat) / assim_tau_s
            fmult = max(mult_hat, 0.5)
        leak_c = F["f_leak_coef"][i] if F else 0.0
        level = V / p.tank_area_m2
        h = solve_hydraulics(p, speed, S["valve_open_cmd"][i], fmult, leak_c, level)

        # first-order lag so step changes are not instantaneous
        if flow_state is None:
            flow_state = dict(h)
        a = 1 - np.exp(-1.0 / tau_flow)
        for k in h:
            flow_state[k] += a * (h[k] - flow_state[k])
        q = max(flow_state["flow_Lpm"], 0.0) * LPM

        # ---- turbidity: completely mixed tank + filter ----
        v_ratio = q / q_nom if q_nom > 0 else 1.0
        eta = np.clip(p.removal_eff * (1 - 0.25 * (v_ratio - 1)), 0.3, 0.97)
        C_in = Ct
        C_out = C_in * (1 - eta)
        dose = S["dose_NTU"][i] + (F["f_dose_NTU"][i] if F else 0.0)
        Ct = Ct + (q / V) * (C_out - Ct) + r_bg * (V0 / V) + dose
        M += q * (C_in - C_out) * NTU_TO_MGL  # g/s (m3/s * g/m3)

        # ---- water lost to the leak ----
        V = max(V - flow_state["leak_Lpm"] * LPM, 0.3 * V0)

        # ---- temperature ----
        T_lag += (S["pump_speed_cmd"][i] * 0.6 - T_lag) / 1800.0  # pump heat
        T_true = p.temp_base_C + 0.8 * np.sin(2 * np.pi * (t + t_offset) / 86400) + T_lag
        if measured_temp is not None:  # twin mode: temperature is measured, not modeled
            T_true = measured_temp[i]
        T_use = T_true

        # ---- dissolved oxygen ----
        Cs = float(do_saturation(T_use))
        if Cdo is None:
            Cdo = Cs - 0.4
        aer = int(S["aeration_cmd"][i])
        kla = p.kla_per_h[aer] * (F["f_kla_mult"][i] if F else 1.0) + p.kla_splash_per_h * v_ratio
        Cdo += (kla / 3600) * (Cs - Cdo) - p.o2_demand_mgL_h / 3600
        Cdo = max(Cdo, 0.0)

        # ---- pH: CO2 stripping by aeration ----
        ph_eq = p.ph_base + p.ph_aeration_gain * (p.kla_per_h[aer] * (F["f_kla_mult"][i] if F else 1.0)) / max(p.kla_per_h)
        ph += (ph_eq - ph) / p.ph_tau_s

        out["flow_Lpm"][i] = flow_state["flow_Lpm"]
        out["q_pump_Lpm"][i] = flow_state["q_pump_Lpm"]
        out["leak_Lpm"][i] = flow_state["leak_Lpm"]
        out["press_kPa"][i] = flow_state["press_kPa"]
        out["dp_filter_kPa"][i] = flow_state["dp_filter_kPa"]
        out["level_cm"][i] = level * 100
        out["turb_in_NTU"][i] = C_in
        out["turb_out_NTU"][i] = C_out
        out["do_mgL"][i] = Cdo
        out["temp_C"][i] = T_true
        out["ph"][i] = ph
        out["cond_uScm"][i] = p.cond_base_uScm * (1 + 0.02 * (T_true - 25)) + p.cond_per_NTU * Ct
        out["filter_mult"][i] = fmult
    return out
