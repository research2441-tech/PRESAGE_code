"""Metrics, conformal lead-time lower bounds, weighted conformal under shift and conformal risk control."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

H, K, W = 21, 3, 14


def cindex_cr(risk, delta, cause, j, horizon=H):
    """truncated cause-specific concordance for competing risks (Wolbers-type comparability)"""
    ev = (cause == j) & (delta <= horizon)
    idx = np.where(ev)[0]
    if len(idx) == 0:
        return np.nan
    num = den = 0.0
    for i in idx:
        comp = (delta > delta[i]) | ((cause != j) & (cause >= 0) & (delta <= delta[i]))
        comp[i] = False
        r = risk[comp]
        den += len(r)
        num += (risk[i] > r).sum() + 0.5 * (risk[i] == r).sum()
    return num / den if den else np.nan


def brier_ibs(cif, delta, cause):
    """integrated (over 1..H) cause-specific Brier score; averaged over causes"""
    out = []
    keep = ~((cause == -2) & (delta < H))
    cif, delta, cause = cif[keep], delta[keep], cause[keep]
    for j in range(cif.shape[2]):
        bs = []
        for h in range(1, H + 1):
            y = ((cause == j) & (delta <= h)).astype(float)
            bs.append(np.mean((y - cif[:, h - 1, j]) ** 2))
        out.append(np.mean(bs))
    return float(np.mean(out)), out


def ece_binary(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if s.any():
            e += s.mean() * abs(p[s].mean() - y[s].mean())
    return e


def point_metrics(pred, window=W):
    cif, d, c = pred["cif"], pred["delta"], pred["cause"]
    cidx = [cindex_cr(cif[:, H - 1, j], d, c, j) for j in range(cif.shape[2])]
    ibs, _ = brier_ibs(cif, d, c)
    ev = (c >= 0) & (d <= H)
    f1 = f1_score(c[ev], cif[ev, H - 1].argmax(1), average="macro") if ev.any() else np.nan
    pw = cif[:, window - 1].sum(1); yw = ((c >= 0) & (d <= window)).astype(float)
    return dict(cindex=float(np.nanmean(cidx)), cindex_by_cause=[float(x) for x in cidx], ibs=ibs,
                f1=float(f1), ece=float(ece_binary(pw, yw)))


# --------------------------------------------------------------------------- alerts + CRC
def season_alert_table(pred, seasons, risk, theta, window=W, targets=(0, 1, 2)):
    """per-season: hit (alert within [T-window, T)), first alert day in window, false alert count."""
    out = {}
    tg = np.array(targets)
    for s in np.unique(pred["season"]):
        sel = pred["season"] == s
        lm = pred["lm"][sel]; r = risk[sel]; d = pred["delta"][sel]; c = pred["cause"][sel]
        alert = r >= theta
        ev = bool(np.isin(c[0], tg))
        T = lm[0] + d[0] if ev else None
        inwin = (d <= window) & np.isin(c, tg)
        hit = bool((alert & inwin).any()) if ev else None
        first = lm[alert & inwin].min() if ev and hit else None
        fa = int((alert & ~inwin).sum())
        out[int(s)] = dict(ev=ev, T=T, hit=hit, first=first, fa=fa, n=int(sel.sum()), alerts=int(alert.sum()),
                           has_window=bool(inwin.any()))
    return out


def crc_threshold(pred, seasons, risk, alpha, grid=None, window=W, targets=(0, 1, 2)):
    """conformal risk control: largest threshold whose (n R + 1)/(n + 1) bound on missed onsets is <= alpha"""
    grid = np.unique(np.quantile(risk, np.linspace(0, 1, 400))) if grid is None else grid
    sid = np.unique(pred["season"]); n = len(sid)
    best = grid.min()
    for th in np.sort(grid)[::-1]:
        tab = season_alert_table(pred, seasons, risk, th, window, targets)
        miss = np.mean([1.0 if (v["ev"] and v["has_window"] and not v["hit"]) else 0.0 for v in tab.values()])
        if (n * miss + 1) / (n + 1) <= alpha:
            best = th
            break
    return float(best)


def alert_metrics(pred, seasons, risk, theta, cause_pred=None, window=W, targets=(0, 1, 2)):
    tab = season_alert_table(pred, seasons, risk, theta, window, targets)
    evs = [s for s, v in tab.items() if v["ev"] and v["has_window"]]
    miss = np.mean([0.0 if tab[s]["hit"] else 1.0 for s in evs])
    leads_on, leads_vis, cause_ok = [], [], []
    for s in evs:
        v = tab[s]
        if not v["hit"]:
            continue
        S = seasons[s]; J = S["J"]
        leads_on.append(v["T"] - v["first"])
        if "vis" in S and S["vis"][J] is not None:
            leads_vis.append(S["vis"][J] - v["first"])
        if cause_pred is not None and len(targets) > 1:
            sel = (pred["season"] == s) & (pred["lm"] == v["first"])
            cause_ok.append((J, int(cause_pred[sel][0])))
    tot_alerts = sum(v["alerts"] for v in tab.values()); tot_fa = sum(v["fa"] for v in tab.values())
    tot_lm = sum(v["n"] for v in tab.values())
    res = dict(miss=float(miss), lead_onset=float(np.median(leads_on)) if leads_on else np.nan,
               lead_visible=float(np.median(leads_vis)) if leads_vis else np.nan,
               lead_visible_mean=float(np.mean(leads_vis)) if leads_vis else np.nan,
               false_alert_frac=float(tot_fa / max(tot_alerts, 1)), fa_per_100lm=float(100 * tot_fa / tot_lm),
               alert_rate=float(tot_alerts / tot_lm))
    if cause_ok:
        a = np.array(cause_ok)
        res["alert_cause_f1"] = float(f1_score(a[:, 0], a[:, 1], average="macro"))
    return res, leads_vis


# --------------------------------------------------------------------------- lead-time lower bounds
def base_lower(cif, a0=0.1):
    """smallest horizon at which the any-cause CIF reaches a0 (H+1 when never reached)"""
    tot = cif.sum(2)
    hit = tot >= a0
    first = np.where(hit.any(1), hit.argmax(1) + 1, H + 1)
    return first.astype(float)


def one_per_season(pred, rng):
    idx = []
    for s in np.unique(pred["season"]):
        w = np.where(pred["season"] == s)[0]
        idx.append(rng.choice(w))
    return np.array(idx)


def weighted_quantile(scores, w_cal, w_test, q):
    """Tibshirani et al. weighted conformal quantile with the test point's weight placed at +inf"""
    o = np.argsort(scores); s = scores[o]; wc = w_cal[o]
    tot = wc.sum() + w_test
    cdf = np.cumsum(wc) / tot
    k = np.searchsorted(cdf, q, side="left")
    return s[k] if k < len(s) else np.inf


def lpb_calibrate(cal, alpha=0.1, a0=0.1, seed=0, n_rep=20):
    """returns the split-conformal correction (median over repeated one-landmark-per-season draws)"""
    rng = np.random.default_rng(seed)
    qs = []
    for _ in range(n_rep):
        idx = one_per_season(cal, rng)
        q = base_lower(cal["cif"][idx], a0)
        y = np.minimum(cal["delta"][idx], H + 1).astype(float)
        sc = q - y
        n = len(sc); k = int(np.ceil((n + 1) * (1 - alpha)))
        qs.append(np.sort(sc)[min(k, n) - 1])
    return float(np.median(qs))


def lpb_weighted(cal, test, f_cal, f_test, alpha=0.1, a0=0.1, seed=0, n_rep=10):
    """covariate-shift weighted conformal lower bounds; density ratio from a probabilistic domain classifier"""
    rng = np.random.default_rng(seed)
    X = np.vstack([f_cal, f_test]); yd = np.r_[np.zeros(len(f_cal)), np.ones(len(f_test))]
    clf = LogisticRegression(max_iter=2000, C=0.5).fit(X, yd)
    pc = np.clip(clf.predict_proba(f_cal)[:, 1], 1e-3, 1 - 1e-3); pt = np.clip(clf.predict_proba(f_test)[:, 1], 1e-3, 1 - 1e-3)
    wc_all = pc / (1 - pc); wt_all = pt / (1 - pt)
    cap = 20.0 * np.median(wc_all)  # truncated density ratios (limits the influence of non-overlapping regions)
    wc_all = np.minimum(wc_all, cap); wt_all = np.minimum(wt_all, cap)
    qb_test = base_lower(test["cif"], a0)
    corr = np.zeros((n_rep, len(f_test)))
    for r in range(n_rep):
        idx = one_per_season(cal, rng)
        sc = base_lower(cal["cif"][idx], a0) - np.minimum(cal["delta"][idx], H + 1)
        wc = wc_all[idx]
        o = np.argsort(sc); s = sc[o]; w = wc[o]; cw = np.cumsum(w)
        tot = cw[-1] + wt_all
        # per test point quantile: smallest s with cumulative weight >= (1-alpha) * total
        need = (1 - alpha) * tot
        k = np.searchsorted(cw, need, side="left")
        corr[r] = np.where(k < len(s), s[np.minimum(k, len(s) - 1)], np.inf)
    c = np.median(corr, 0)
    return qb_test - c, wt_all


def lpb_eval(lpb, test):
    y = np.minimum(test["delta"], 10 ** 4)
    known = ~((test["cause"] == -2) & (y < lpb))  # censored before the bound: coverage unknown
    cov = float(np.mean((y >= lpb)[known]))
    ev = test["cause"] >= 0
    cov_ev = float(np.mean(y[ev & (test["delta"] <= H)] >= lpb[ev & (test["delta"] <= H)]))
    finite = np.clip(lpb, 0, H + 1)
    return dict(coverage=cov, coverage_events=cov_ev, mean_lpb=float(np.mean(finite)),
                informative=float(np.mean(finite >= 3)))
