"""Builds the real-weather-driven multimodal crop stress onset benchmark (FVG stations).

Each field-season = (station, year, management scenario). WOFOST 8.1 (water + N limited) provides crop
physiology; a compartment epidemic provides latent / visible infection; sensor models then generate
asynchronous observation tokens. Output: data/sim_fieldseasons.pkl
"""
import datetime as dt
import glob
import math
import os
import pickle
import sys
import warnings
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from sim_wofost import FrameWeather, run_season, crop_provider, SOILS  # noqa
from disease import epidemic  # noqa

FVG = "./data/fvg"
OUT = "./presage/data"
REQ = ["rr", "t180_min", "t180_max", "rh_med", "v10_med", "rg"]
N_SCEN = int(os.environ.get("N_SCEN", "6"))
KAPPA_BIO = float(os.environ.get("KAPPA_BIO", "1.0"))  # strength of pre-symptomatic signatures of latent infection

MOD = {"static": 0, "weather": 1, "soil": 2, "optical": 3, "thermal": 4, "mgmt": 5, "trap": 6}
TOD = {"static": 0.0, "mgmt": 0.30, "trap": 0.40, "optical": 0.45, "soil": 0.50, "thermal": 0.55, "weather": 0.99}


def zone_of(alt, lat):
    if alt < 15:
        return 0  # coastal lowland
    if alt < 120:
        return 1  # alluvial plain
    if alt < 400:
        return 2  # foothill / morainic hills
    return 3  # alpine valley


def station_years():
    st = pd.read_parquet(f"{FVG}/stations.parquet").set_index("code")
    items = []
    for f in sorted(glob.glob(f"{FVG}/daily/*.parquet")):
        d = pd.read_parquet(f)
        code = d.code.iloc[0]
        if code not in st.index:
            continue
        d["date"] = pd.to_datetime(d["date"])
        for yr, g in d.groupby(d.date.dt.year):
            gs = g[(g.date.dt.month >= 4) & (g.date.dt.month <= 9)]
            if len(gs) < 170:
                continue
            if gs[REQ].notna().all(axis=1).mean() < 0.95:
                continue
            w = g[(g.date >= f"{yr}-03-01") & (g.date <= f"{yr}-11-30")].copy()
            if len(w) < 260:
                continue
            items.append((code, int(yr), float(st.loc[code, "alt"]), float(st.loc[code, "lat"]),
                          float(st.loc[code, "lon"]), w))
    return items


def vpd_kpa(tmin, tmax, rh):
    es = 0.5 * (0.6108 * np.exp(17.27 * tmax / (tmax + 237.3)) + 0.6108 * np.exp(17.27 * tmin / (tmin + 237.3)))
    return np.clip(es * (1 - rh / 100.), 0.05, 5.0)


def sustained(mask, k):
    """first index i where mask[i:i+k] all True"""
    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= k:
            return i - k + 1
    return None


def simulate(args):
    code, yr, alt, lat, lon, w, scen = args
    rng = np.random.default_rng(zlib.crc32(f"{code}|{yr}|{scen}|{KAPPA_BIO}".encode()))
    w = w.rename(columns={"t180_min": "tmin", "t180_max": "tmax", "rh_med": "rh", "v10_med": "wind", "rr": "rain"})
    num = ["tmin", "tmax", "rh", "wind", "rain", "rg"]
    w.loc[(w.rg > 34000) | (w.rg < 0), "rg"] = np.nan  # quality control of implausible records
    w.loc[(w.tmax > 42) | (w.tmax < -25), "tmax"] = np.nan
    w.loc[(w.tmin > 32) | (w.tmin < -30), "tmin"] = np.nan
    w.loc[(w.rh > 100) | (w.rh < 5), "rh"] = np.nan
    w.loc[(w.rain < 0) | (w.rain > 240), "rain"] = np.nan
    w.loc[(w.wind < 0) | (w.wind > 120), "wind"] = np.nan
    miss = w[num].isna().any(axis=1).values
    w[num] = w[num].astype(float).interpolate(limit_direction="both")
    w = w.reset_index(drop=True)
    try:
        wdp = FrameWeather(w, lat, lon, alt)
    except Exception:
        return None
    crop = "maize" if rng.random() < 0.6 else "soybean"
    variety = "Grain_maize_201" if crop == "maize" else "Soybean_901"
    soil = list(SOILS)[rng.integers(3)]
    if crop == "maize":
        sow = dt.date(yr, 4, 5) + dt.timedelta(days=int(rng.integers(0, 35)))
    else:
        sow = dt.date(yr, 4, 25) + dt.timedelta(days=int(rng.integers(0, 35)))
    nsoil = float(rng.uniform(15, 90))
    n_events = []
    if crop == "maize":
        plan = rng.choice(["none", "low", "split"], p=[0.25, 0.35, 0.40])
        if plan == "low":
            n_events = [(sow + dt.timedelta(days=int(rng.integers(20, 40))), float(rng.uniform(30, 70)))]
        elif plan == "split":
            n_events = [(sow + dt.timedelta(days=int(rng.integers(15, 30))), float(rng.uniform(40, 80))),
                        (sow + dt.timedelta(days=int(rng.integers(45, 60))), float(rng.uniform(40, 90)))]
    else:
        if rng.random() < 0.3:
            n_events = [(sow + dt.timedelta(days=int(rng.integers(15, 30))), float(rng.uniform(15, 35)))]
    irr = []
    irr_plan = rng.choice(["rainfed", "calendar", "late"], p=[0.45, 0.35, 0.20])
    if irr_plan != "rainfed":
        start = sow + dt.timedelta(days=int(rng.integers(35, 55) if irr_plan == "calendar" else rng.integers(60, 80)))
        step = int(rng.integers(7, 15))
        amt = float(rng.uniform(1.5, 3.5))  # cm
        d = start
        while d < sow + dt.timedelta(days=130):
            irr.append((d, amt))
            d += dt.timedelta(days=step)
    wav = float(rng.uniform(4, 16))
    try:
        out, cp = run_season(wdp, crop, variety, sow, soil, n_events, irr, nsoil, wav)
    except Exception as e:  # pragma: no cover
        return None
    out["day"] = pd.to_datetime(out["day"])
    out = out[out.DVS.notna()].reset_index(drop=True)
    if len(out) < 60:
        return None
    tb = cp["NMAXLV_TB"]
    xs, ys = tb[0::2], tb[1::2]
    nmaxlv = np.interp(out.DVS.values, xs, ys)
    with np.errstate(divide="ignore", invalid="ignore"):
        ns = np.where(out.WLV.values > 1, out.NamountLV.values / (out.WLV.values * nmaxlv), 1.0)
    ns = np.nan_to_num(ns, nan=1.0)
    # align weather to crop days
    wd = w.set_index(pd.to_datetime(w.date))
    days = out.day
    tavg = 0.5 * (wd.loc[days, "tmin"].values + wd.loc[days, "tmax"].values)
    rain = wd.loc[days, "rain"].values
    rh = wd.loc[days, "rh"].values
    dvs = out.DVS.values
    n = len(out)
    arrival = int(rng.integers(15, min(85, n - 10))) if rng.random() < 0.75 else None
    resist = float(rng.uniform(0.0, 0.6))
    L, S, pressure = epidemic(tavg, rain, rh, dvs, arrival, resist, crop, rng)
    rftra = np.nan_to_num(out.RFTRA.values, nan=1.0)
    lai = np.nan_to_num(out.LAI.values, nan=0.0)
    sm = out.SM.values
    # ---------- onsets ----------
    valid = dvs < 1.6
    end = int(np.argmax(~valid)) if (~valid).any() else n
    start = 20
    def first(mask, k, lo=start):
        m = mask.copy(); m[:lo] = False; m[end:] = False
        return sustained(m, k)
    Tw = first((rftra < 0.8) & (lai > 0.6), 2)
    Tn = first((ns < 0.75) & (dvs > 0.25), 3)
    Tp = first((L + S) >= 4e-3, 1)
    Vw = first((rftra < 0.45) & (lai > 0.6), 2, lo=Tw if Tw is not None else start)
    Vn = first((ns < 0.55) & (dvs > 0.25), 2, lo=Tn if Tn is not None else start)
    Vp = first(S >= 0.01, 1, lo=Tp if Tp is not None else start)
    onsets = {0: Tw, 1: Tn, 2: Tp}
    vis = {0: Vw, 1: Vn, 2: Vp}
    cand = [(t, j) for j, t in onsets.items() if t is not None]
    if cand:
        T, J = min(cand)
    else:
        T, J = None, -1
    # ---------- sensor tokens ----------
    toks = []  # (t, modality, values list)
    tmin_ = wd.loc[days, "tmin"].values; tmax_ = wd.loc[days, "tmax"].values
    rg = wd.loc[days, "rg"].values; wind = wd.loc[days, "wind"].values
    vpd = vpd_kpa(tmin_, tmax_, rh)
    gdd = np.cumsum(np.clip(tavg - (10 if crop == "maize" else 8), 0, None))
    dmiss = pd.Series(miss, index=pd.to_datetime(w.date)).loc[days].values
    soilp = SOILS[soil]
    toks.append((0.0, MOD["static"], [1.0 if crop == "maize" else 0.0, soilp["SMW"], soilp["SMFCF"], soilp["SM0"],
                                       alt / 1000., lat - 46.0]))
    for d in range(n):
        if not dmiss[d]:
            toks.append((d + TOD["weather"], MOD["weather"], [tmin_[d], tmax_[d], rh[d], math.log1p(rain[d]), rg[d] / 1000.,
                                                              wind[d], vpd[d], gdd[d] / 1000.]))
    # soil moisture probe with bias and failure episodes
    bias = rng.normal(0, 0.02)
    dead = 0
    for d in range(n):
        if dead > 0:
            dead -= 1; continue
        if rng.random() < 0.012:
            dead = int(rng.geometric(1 / 9.)); continue
        toks.append((d + TOD["soil"], MOD["soil"], [sm[d] + bias + rng.normal(0, 0.008)]))
    # optical revisit (5-day) with clouds
    phase = int(rng.integers(0, 5))
    Lrel = np.clip(L / 0.01, 0, 1)
    for d in range(phase, n, 5):
        cloudy = rain[d] > 0.5 or (rh[d] > 85 and rng.random() < 0.6) or rng.random() < 0.12
        if cloudy:
            continue
        roll = 1.0 - 0.5 * max(0.0, 0.5 - rftra[d])
        laig = lai[d] * (1 - S[d]) * roll
        fc = 1 - math.exp(-0.55 * laig)
        chl = (np.clip(ns[d], 0.3, 1.1) - 0.3) / 0.8
        ndvi = 0.12 + 0.78 * fc + rng.normal(0, 0.02)
        ndre = 0.04 + (1 - math.exp(-0.5 * laig)) * (0.12 + 0.38 * chl) - KAPPA_BIO * 0.03 * Lrel[d] + rng.normal(0, 0.012)
        toks.append((d + TOD["optical"], MOD["optical"], [ndvi, ndre]))
    # UAV thermal flights
    d = int(rng.integers(0, 4))
    while d < n:
        if rain[d] < 0.5 and rg[d] / 1000. > 12:
            fc = 1 - math.exp(-0.55 * lai[d])
            vp = vpd[d]
            ll = 2.0 - 1.9 * vp
            ul = 4.5
            dT = ll + (1 - rftra[d]) * (ul - ll) + KAPPA_BIO * 1.2 * Lrel[d] + (1 - fc) * (3.0 + rng.normal(0, 1.0)) + rng.normal(0, 0.5)
            toks.append((d + TOD["thermal"], MOD["thermal"], [dT, tmax_[d], vp]))
            d += int(rng.integers(3, 8))
        else:
            d += 1
    # regional spore-trap counts (weekly), informative about inoculum arrival but noisy
    wk = int(rng.integers(0, 7))
    for d in range(wk, n, 7):
        lam = 0.15
        if arrival is not None and d >= arrival - 3:
            lam = 1.5 + 40.0 * float(L[d] + S[d])
        toks.append((d + TOD["trap"], MOD["trap"], [math.log1p(rng.poisson(lam))]))
    for ddate, a in n_events:
        k = (pd.Timestamp(ddate) - days.iloc[0]).days
        if 0 <= k < n:
            toks.append((k + TOD["mgmt"], MOD["mgmt"], [a / 100., 0.0]))
    for ddate, a in irr:
        k = (pd.Timestamp(ddate) - days.iloc[0]).days
        if 0 <= k < n:
            toks.append((k + TOD["mgmt"], MOD["mgmt"], [0.0, a / 3.0]))
    toks.sort(key=lambda x: (x[0], x[1]))
    # landmarks
    stop = T if T is not None else end
    lms = list(range(start, max(start, stop), 3))
    return dict(code=code, year=yr, alt=alt, lat=lat, zone=zone_of(alt, lat), scen=scen, crop=crop, soil=soil,
                n=n, end=end, T=T, J=J, onsets=onsets, vis=vis, landmarks=lms,
                tok_t=np.array([x[0] for x in toks], dtype=np.float32),
                tok_m=np.array([x[1] for x in toks], dtype=np.int64),
                tok_v=[np.array(x[2], dtype=np.float32) for x in toks],
                truth=dict(rftra=rftra.astype(np.float32), ns=ns.astype(np.float32), L=L.astype(np.float32),
                           S=S.astype(np.float32), lai=lai.astype(np.float32), sm=sm.astype(np.float32),
                           pressure=pressure.astype(np.float32), dvs=dvs.astype(np.float32)),
                irr_plan=str(irr_plan), nplan=len(n_events))


def main():
    os.makedirs(OUT, exist_ok=True)
    items = station_years()
    print("station-years", len(items), "stations", len({i[0] for i in items}))
    crop_provider()
    jobs = [(c, y, a, la, lo, w, s) for (c, y, a, la, lo, w) in items for s in range(N_SCEN)]
    with Pool(2) as pool:
        res = pool.map(simulate, jobs, chunksize=8)
    res = [r for r in res if r is not None]
    tag = os.environ.get("TAG", "")
    with open(f"{OUT}/sim_fieldseasons{tag}.pkl", "wb") as f:
        pickle.dump(res, f)
    df = pd.DataFrame([dict(zone=r["zone"], crop=r["crop"], J=r["J"], T=r["T"], year=r["year"],
                            lead=(r["vis"][r["J"]] - r["T"]) if r["J"] >= 0 and r["vis"][r["J"]] is not None else None,
                            ntok=len(r["tok_t"]), nlm=len(r["landmarks"])) for r in res])
    print(len(df)); print(df.J.value_counts()); print(df.groupby("zone").J.value_counts().unstack())
    print(df.groupby("J").lead.describe()); print(df.ntok.describe()); print(df.nlm.sum())


if __name__ == "__main__":
    main()
