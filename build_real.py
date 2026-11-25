"""Real-field external benchmark from the FBK soil-moisture dataset (Trentino, three consortia).

Unit = dry-down episode of one tensiometer stream. Competing events: soil-water stress onset (tension
reaching U hPa) versus rewetting (tension returning below L0 after exceeding 200 hPa) before stress.
Asynchronous modalities: gridded weather (daily), tensiometer (daily with gaps), satellite indices on
acquisition days only, irrigation meter events, static soil descriptors."""
import os
import pickle
import re
import numpy as np
import pandas as pd

D = "./data/it4lia/"
OUT = "./presage/data"
U, L0, MID = 300.0, 150.0, 200.0
STEP = 2


def load(name, c):
    f = f"{D}{name}_consortium{c}.parquet"
    if not os.path.exists(f):
        return None
    df = pd.read_parquet(f)
    if "result_time" not in df.columns:
        df = df.reset_index()
    if "result_time" in df.columns:
        df["result_time"] = pd.to_datetime(df["result_time"])
    return df


def weather(c):
    hw = load("historical_weather_data", c)
    cols = ["temperature_2m_min", "temperature_2m_max", "relative_humidity_2m", "precipitation",
            "direct_radiation", "wind_speed_10m", "et0_fao_evapotranspiration", "soil_moisture_0_to_7cm"]
    w = hw.groupby("result_time")[cols].median().sort_index()
    return w


def acquisitions(g, col):
    """recover satellite acquisition days from a daily, linearly interpolated index series (slope breaks)"""
    v = g[col].values
    d2 = np.abs(np.diff(v, 2))
    idx = np.where(d2 > 1e-6)[0] + 1
    return g.iloc[idx]


def rs_series(c):
    rs = load("remote_sensing_data_final", c)
    if rs is None:
        return {}
    rs = rs.replace([np.inf, -np.inf], np.nan)
    out = {}
    for name, g in rs.groupby("datastream_name"):
        g = g.sort_values("result_time").dropna(subset=["ndvi", "ndmi"])
        a = acquisitions(g, "ndvi")
        out[name] = a[["result_time", "ndvi", "ndmi"]].reset_index(drop=True)
    return out


def sector_of(name):
    m = re.search(r"sector(\d+)", name)
    return m.group(1) if m else None


def build():
    units = []
    for c in range(3):
        W = weather(c)
        sens = load("field_sensor_data", c)
        sens = sens[sens.sensor_type == "Soil Moisture Tension"]
        irr = load("irrigation_data", c)
        soil = load("soil_type_data", c)
        rs = rs_series(c)
        for name, g in sens.groupby("datastream_name"):
            g = g.sort_values("result_time")
            depth = -float(g.ground_offset.iloc[0])
            # --- joins
            rs_k, irr_k, soil_row = [], [], None
            if c == 1:
                rs_k = [k for k in rs if k == name]
                if soil is not None:
                    sr = soil[soil.datastream_name == name]
                    soil_row = sr.iloc[0] if len(sr) else None
            elif c == 0:
                sec = sector_of(name)
                cand = [k for k in rs if sector_of(k) == sec]
                rs_k = sorted(cand, key=lambda k: -len(rs[k]))[:1]  # one representative stream per sector
                irr_k = [k for k in irr.datastream_name.unique() if sector_of(k) == sec]
                if soil is not None:
                    sr = soil[[sector_of(x) == sec for x in soil.datastream_name]]
                    soil_row = sr.iloc[0] if len(sr) else None
            else:
                mm = re.search(r"management(\d)_(\d+)$", name)
                if mm:
                    irr_k = [f"consortium2_Contatore_management{mm.group(1)}_{mm.group(2)}"]
            for yr in sorted(g.result_time.dt.year.unique()):
                t0 = pd.Timestamp(f"{yr}-04-01"); t1 = pd.Timestamp(f"{yr}-10-31")
                gy = g[(g.result_time >= t0) & (g.result_time <= t1)]
                if len(gy) < 30:
                    continue
                day = lambda ts: (ts - t0).days
                toks = []
                st = [depth / 100.]
                if soil_row is not None:
                    st += [float(soil_row["Sand"]) / 100, float(soil_row["Silt"]) / 100, float(soil_row["Clay"]) / 100,
                           float(soil_row["RZD_Modal_(cm)"]) / 100 if not pd.isna(soil_row["RZD_Modal_(cm)"]) else 0.5, 1.0]
                else:
                    st += [0, 0, 0, 0, 0.0]
                toks.append((0.0, 0, st))
                wy = W[(W.index >= t0) & (W.index <= t1)]
                for ts, r in wy.iterrows():
                    toks.append((day(ts) + 0.99, 1, [r.iloc[0], r.iloc[1], r.iloc[2], np.log1p(max(r.iloc[3], 0)),
                                                     r.iloc[4] / 100., r.iloc[5], r.iloc[6], r.iloc[7]]))
                for ts, v in zip(gy.result_time, gy.result.values):
                    if not np.isnan(v):
                        toks.append((day(ts) + 0.5, 2, [np.log1p(max(v, 0)) / np.log(1000.)]))
                for k in rs_k:
                    a = rs[k]; a = a[(a.result_time >= t0) & (a.result_time <= t1)]
                    for ts, nd, nm in zip(a.result_time, a.ndvi, a.ndmi):
                        toks.append((day(ts) + 0.45, 3, [nd, nm]))
                if irr_k:
                    ir = irr[irr.datastream_name.isin(irr_k)]
                    ir = ir[(ir.result_time >= t0) & (ir.result_time <= t1)].groupby("result_time").result.sum()
                    for ts, v in ir.items():
                        if v > 0:
                            toks.append((day(ts) + 0.30, 5, [0.0, float(np.log1p(v))]))
                toks.sort(key=lambda x: (x[0], x[1]))
                # --- episodes (state machine over daily tension)
                vals = gy.set_index(gy.result_time.map(day)).result
                vals = vals[~vals.index.duplicated()]
                days = np.arange(0, (t1 - t0).days + 1)
                ser = vals.reindex(days)
                state = "wet" if (len(vals) and vals.iloc[0] <= L0) else "dry"
                b, peak = (0 if state == "wet" else None), 0.0
                for d in days:
                    x = ser.loc[d]
                    if np.isnan(x):
                        continue
                    if state == "dry":
                        if x <= L0:
                            state, b, peak = "wet", d, x
                        continue
                    peak = max(peak, x)
                    ev = None
                    if x >= U:
                        ev = 0
                    elif peak >= MID and x <= L0:
                        ev = 1
                    if ev is not None:
                        lms = list(range(b, d, STEP))
                        lms = [l for l in lms if not np.isnan(ser.loc[l])] if lms else []
                        if lms:
                            units.append(dict(code=name, cons=c, year=int(yr), T=int(d), J=int(ev), landmarks=lms,
                                              cens=None, **pack(toks)))
                        if ev == 0:
                            state = "dry"
                        else:
                            state, b, peak = "wet", d, x
                if state == "wet" and b is not None:
                    last = int(vals.index.max())
                    lms = [l for l in range(b, last, STEP) if not np.isnan(ser.loc[l])]
                    if lms:
                        units.append(dict(code=name, cons=c, year=int(yr), T=None, J=-1, landmarks=lms,
                                          cens=last, **pack(toks)))
    return units


def pack(toks):
    return dict(tok_t=np.array([x[0] for x in toks], np.float32), tok_m=np.array([x[1] for x in toks], np.int64),
                tok_v=[np.array(x[2], np.float32) for x in toks])


if __name__ == "__main__":
    u = build()
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(u, open(f"{OUT}/real_episodes.pkl", "wb"))
    df = pd.DataFrame([dict(c=x["cons"], J=x["J"], nlm=len(x["landmarks"]), ntok=len(x["tok_t"]),
                            nrs=int((x["tok_m"] == 3).sum()), nirr=int((x["tok_m"] == 5).sum())) for x in u])
    print(len(df)); print(df.groupby("c").J.value_counts().unstack()); print(df.groupby("c")[["nlm", "ntok", "nrs", "nirr"]].mean())
    print(df.nlm.sum())
