"""Weather-driven foliar epidemic with latent and infectious compartments (daily step).

Structure follows the site-based compartment logic of EPIRICE-type models: healthy (H), latent (L),
infectious (I) and removed (R) leaf-area fractions, with infection efficiency modulated by a
temperature response and a leaf-wetness proxy, and by host age and cultivar resistance.
"""
import numpy as np

PATHO = {
    # crop: Tmin, Topt, Tmax, latent period (d), infectious period (d), basic daily rate
    "maize": dict(tmin=10., topt=24., tmax=32., p=9, i=14, rc=1.3),
    "soybean": dict(tmin=12., topt=25., tmax=33., p=8, i=12, rc=1.3),
}


def temp_response(t, tmin, topt, tmax):
    t = np.clip(t, tmin, tmax)
    a = (t - tmin) / (topt - tmin)
    b = (tmax - t) / (tmax - topt)
    return np.clip(a ** 1.0 * b ** ((tmax - topt) / (topt - tmin)), 0, 1)


def wetness(rain, rh):
    """Daily leaf-wetness proxy in [0, 1] from precipitation and mean relative humidity."""
    w = np.clip((rh - 70.) / 20., 0., 1.) * 0.7
    w = np.where(rain >= 1.0, 1.0, w)
    return w


def epidemic(tavg, rain, rh, dvs, arrival, resist, crop, rng):
    """Returns daily latent fraction L, visible severity S (I+R) and the infection-pressure index."""
    p = PATHO[crop]
    n = len(tavg)
    fT = temp_response(tavg, p["tmin"], p["topt"], p["tmax"])
    fW = wetness(rain, rh)
    # host age: susceptibility rises after canopy closure, declines at senescence
    fA = np.clip(np.where(dvs < 0.3, dvs / 0.3 * 0.5, np.where(dvs < 1.4, 1.0, 1.0 - (dvs - 1.4))), 0.1, 1.0)
    pressure = fT * fW
    lat = np.zeros(n + p["p"] + p["i"] + 2)
    inf = np.zeros_like(lat)
    L = np.zeros(n); S = np.zeros(n)
    H = 1.0; Itot = 0.0; R = 0.0
    queue_lat = []  # (day_becomes_infectious, amount)
    queue_inf = []  # (day_removed, amount)
    Lcur = 0.0
    for d in range(n):
        if arrival is not None and d == arrival:
            amt = 1e-3
            queue_lat.append((d + p["p"], amt)); Lcur += amt; H -= amt
        # transitions
        for q in [x for x in queue_lat if x[0] == d]:
            queue_lat.remove(q); Lcur -= q[1]; Itot += q[1]; queue_inf.append((d + p["i"], q[1]))
        for q in [x for x in queue_inf if x[0] == d]:
            queue_inf.remove(q); Itot -= q[1]; R += q[1]
        # new infections (logistic limitation by healthy fraction)
        rate = p["rc"] * pressure[d] * fA[d] * (1. - resist)
        new = min(H, rate * Itot * H)
        if new > 0:
            pdays = p["p"] + int(rng.integers(-1, 2))
            queue_lat.append((d + pdays, new)); Lcur += new; H -= new
        L[d] = Lcur; S[d] = Itot + R
    return L, S, pressure
