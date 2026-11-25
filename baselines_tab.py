"""Landmark summary features and tabular / rule baselines."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from presage_lib import season_arrays, Prep, H, K, CompetingRiskHead, cr_nll, cif_from_logits

FEATS = ["maize", "SMW", "SMFCF", "SM0", "alt", "lat", "day", "gdd", "tmax7", "rh7", "rain7", "rain30", "vpd7", "rg7",
         "press7", "soil_last", "soil_d5", "soil_age", "ndvi_last", "ndre_last", "ratio_last", "ndvi_max", "ndvi_d",
         "ratio_d", "opt_age", "cwsi_last", "cwsi_mean2", "th_age", "n_tot", "n_age", "irr_tot", "irr_age", "trap_last",
         "trap_sum21"]


def landmark_features(s, prep, drop_mod=None, keep_mask=None):
    t, m, v = season_arrays(s)
    keep = np.ones(len(t), bool) if keep_mask is None else keep_mask.copy()
    if drop_mod is not None:
        for dm in np.atleast_1d(drop_mod):
            keep &= m != dm
    t, m, v = t[keep], m[keep], v[keep]
    prox = prep.proxies(t, m, v)
    rows = []
    for lm in s["landmarks"]:
        c = t < lm + 1.0
        f = dict(zip(FEATS[:6], v[m == 0][0][:6]))
        f["day"] = lm
        w = c & (m == 1)
        wl = v[w]; wt = t[w]; pw = prox[w, 2]
        f["gdd"] = wl[-1, 7] if len(wl) else np.nan
        r7 = wt >= lm - 6; r30 = wt >= lm - 29
        f["tmax7"] = wl[r7, 1].mean() if r7.any() else np.nan
        f["rh7"] = wl[r7, 2].mean() if r7.any() else np.nan
        f["rain7"] = np.expm1(wl[r7, 3]).sum() if r7.any() else np.nan
        f["rain30"] = np.expm1(wl[r30, 3]).sum() if r30.any() else np.nan
        f["vpd7"] = wl[r7, 6].mean() if r7.any() else np.nan
        f["rg7"] = wl[r7, 4].mean() if r7.any() else np.nan
        f["press7"] = pw[-1] if len(pw) else np.nan
        so = c & (m == 2); sv = v[so, 0]; st = t[so]
        f["soil_last"] = sv[-1] if len(sv) else np.nan
        f["soil_d5"] = (sv[-1] - sv[st <= st[-1] - 5][-1]) if len(sv) and (st <= st[-1] - 5).any() else np.nan
        f["soil_age"] = lm + 1 - st[-1] if len(st) else 99
        op = c & (m == 3); ov = v[op]; ot = t[op]; orat = prox[op, 1]
        f["ndvi_last"] = ov[-1, 0] if len(ov) else np.nan
        f["ndre_last"] = ov[-1, 1] if len(ov) else np.nan
        f["ratio_last"] = orat[-1] if len(ov) else np.nan
        f["ndvi_max"] = ov[:, 0].max() if len(ov) else np.nan
        f["ndvi_d"] = ov[-1, 0] - ov[-2, 0] if len(ov) > 1 else np.nan
        f["ratio_d"] = orat[-1] - orat[-2] if len(ov) > 1 else np.nan
        f["opt_age"] = lm + 1 - ot[-1] if len(ot) else 99
        th = c & (m == 4); cw = prox[th, 0]; tt = t[th]
        f["cwsi_last"] = cw[-1] if len(cw) else np.nan
        f["cwsi_mean2"] = cw[-2:].mean() if len(cw) else np.nan
        f["th_age"] = lm + 1 - tt[-1] if len(tt) else 99
        mg = c & (m == 5); mv = v[mg]; mt = t[mg]
        nsel = mv[:, 0] > 0 if len(mv) else np.zeros(0, bool); isel = mv[:, 1] > 0 if len(mv) else np.zeros(0, bool)
        f["n_tot"] = mv[nsel, 0].sum() if len(mv) else 0.0
        f["n_age"] = lm + 1 - mt[nsel][-1] if len(mv) and nsel.any() else 99
        f["irr_tot"] = mv[isel, 1].sum() if len(mv) else 0.0
        f["irr_age"] = lm + 1 - mt[isel][-1] if len(mv) and isel.any() else 99
        tr = c & (m == 6); tv = v[tr, 0]; tt2 = t[tr]
        f["trap_last"] = tv[-1] if len(tv) else np.nan
        f["trap_sum21"] = tv[tt2 >= lm - 20].sum() if len(tv) else np.nan
        rows.append([f[k] for k in FEATS])
    return np.array(rows, np.float32)


def landmark_table(idx, seasons, prep, **kw):
    X, S, LM, D, C = [], [], [], [], []
    for i in idx:
        s = seasons[i]
        if not s["landmarks"]:
            continue
        X.append(landmark_features(s, prep, **kw))
        lms = np.array(s["landmarks"])
        S.append(np.full(len(lms), i)); LM.append(lms)
        if s["T"] is None:
            D.append(np.full(len(lms), 10 ** 4)); C.append(np.full(len(lms), -1))
        else:
            D.append(s["T"] - lms); C.append(np.full(len(lms), s["J"]))
    return dict(X=np.concatenate(X), season=np.concatenate(S), lm=np.concatenate(LM).astype(float),
                delta=np.concatenate(D), cause=np.concatenate(C))


class Imputer:
    def fit(self, X):
        self.mu = np.nanmean(X, 0); self.sd = np.nanstd(X, 0) + 1e-6
        return self

    def __call__(self, X):
        M = np.isnan(X).astype(np.float32)
        Z = (np.where(np.isnan(X), self.mu, X) - self.mu) / self.sd
        return np.hstack([Z, M]).astype(np.float32)


# ------------------------------------------------------------------ rules (agronomic thresholds)
def rule_scores(tab):
    X = tab["X"]; ix = {k: i for i, k in enumerate(FEATS)}
    cw = np.nan_to_num(X[:, ix["cwsi_last"]], nan=0.0)
    ndrop = np.nan_to_num(X[:, ix["ndvi_max"]] - X[:, ix["ndvi_last"]], nan=0.0)
    rat = np.nan_to_num(-X[:, ix["ratio_d"]], nan=0.0)
    pr = np.nan_to_num(X[:, ix["press7"]], nan=0.0)
    soil = np.nan_to_num(X[:, ix["SMFCF"]] - X[:, ix["soil_last"]], nan=0.0) / np.maximum(X[:, ix["SMFCF"]] - X[:, ix["SMW"]], 0.05)
    comp = np.stack([np.maximum((cw - 0.2) / 0.4, (soil - 0.4) / 0.4), np.maximum(rat / 0.08, ndrop / 0.15),
                     pr / 0.5], 1)
    return comp  # per-cause risk scores


# ------------------------------------------------------------------ discrete-time gradient boosting
def hgb_fit(tab, seed=0):
    X, d, c = tab["X"], tab["delta"], tab["cause"]
    rows, ys = [], []
    for i in range(len(X)):
        last = min(d[i], H) if c[i] >= 0 else H
        for h in range(1, last + 1):
            rows.append(np.r_[X[i], h]); ys.append(c[i] + 1 if (c[i] >= 0 and h == d[i]) else 0)
    rows = np.array(rows, np.float32); ys = np.array(ys)
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, random_state=seed,
                                         l2_regularization=1.0)
    clf.fit(rows, ys)
    return clf


def hgb_cif(clf, tab):
    X = tab["X"]; n = len(X)
    probs = np.zeros((n, H, K + 1))
    for h in range(1, H + 1):
        p = clf.predict_proba(np.c_[X, np.full(n, h)])
        full = np.zeros((n, K + 1)); full[:, clf.classes_] = p
        probs[:, h - 1] = full
    surv = np.cumprod(probs[:, :, 0], 1)
    prev = np.concatenate([np.ones((n, 1)), surv[:, :-1]], 1)
    return np.cumsum(prev[:, :, None] * probs[:, :, 1:], 1)


# ------------------------------------------------------------------ cause-specific Cox (lifelines)
def cox_fit(tab, imp):
    from lifelines import CoxPHFitter
    import pandas as pd
    Z = imp(tab["X"])[:, :len(FEATS)]
    d, c = tab["delta"], tab["cause"]
    tt = np.minimum(np.where(c >= 0, d, H), H).astype(float)
    models = []
    for j in range(K):
        e = ((c == j) & (d <= H)).astype(int)
        df = pd.DataFrame(Z, columns=FEATS); df["T"] = tt; df["E"] = e
        df = df.loc[:, df.std() > 1e-6] if True else df
        cph = CoxPHFitter(penalizer=0.1).fit(df, "T", "E")
        models.append(cph)
    return models


def cox_cif(models, tab, imp):
    import pandas as pd
    Z = imp(tab["X"])[:, :len(FEATS)]
    df = pd.DataFrame(Z, columns=FEATS)
    n = len(df); hz = np.zeros((n, H, K))
    for j, m in enumerate(models):
        cols = [c for c in m.params_.index]
        bh = m.baseline_cumulative_hazard_
        grid = np.arange(1, H + 1)
        Lam = np.interp(grid, bh.index.values, bh.values[:, 0])
        dL = np.diff(np.r_[0, Lam])
        risk = np.exp((df[cols].values - m._norm_mean[cols].values) @ m.params_.values)
        hz[:, :, j] = np.clip(risk[:, None] * dL[None, :], 0, 5)
    tot = hz.sum(2)
    surv = np.exp(-np.cumsum(tot, 1))
    prev = np.concatenate([np.ones((n, 1)), surv[:, :-1]], 1)
    frac = np.where(tot[:, :, None] > 0, hz / np.maximum(tot[:, :, None], 1e-12), 0)
    inc = prev[:, :, None] * (1 - np.exp(-tot))[:, :, None] * frac
    return np.cumsum(inc, 1)


# ------------------------------------------------------------------ DeepHit-type MLP and snapshot classifier
class MLPCR(nn.Module):
    def __init__(self, d_in, D=64):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, D), nn.GELU(), nn.Dropout(0.1), nn.Linear(D, D), nn.GELU())
        self.head = CompetingRiskHead(D, anchor=False)

    def forward(self, x):
        return self.head(self.body(x))


def deephit_rank(logits, delta, cause, sigma=0.5):
    cif = cif_from_logits(logits)  # n H K
    loss = 0.0; cnt = 0
    for j in range(K):
        ev = torch.where((cause == j) & (delta <= H))[0]
        if len(ev) == 0:
            continue
        di = delta[ev]
        Fi = cif[ev, di - 1, j]  # own CIF at own time
        Fall = cif[:, :, j][:, (di - 1)]  # n x m
        comp = (delta[:, None] > di[None, :]).float()
        loss = loss + (comp * torch.exp(-(Fi[None, :] - Fall) / sigma)).sum() / comp.sum().clamp(min=1)
        cnt += 1
    return loss / max(cnt, 1)


def train_mlp(model, Xtr, dtr, ctr, Xva, dva, cva, epochs=60, lr=1e-3, seed=0, rank=0.2, snapshot=False):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xtr, Xva = torch.tensor(Xtr), torch.tensor(Xva)
    dtr, ctr, dva, cva = map(lambda a: torch.tensor(a, dtype=torch.long), (dtr, ctr, dva, cva))
    rng = np.random.default_rng(seed); best, state, bad = 1e9, None, 0

    def loss_fn(x, d, c):
        out = model(x)
        if snapshot:
            y = torch.where((c >= 0) & (d <= 14), c + 1, torch.zeros_like(c))
            return F.cross_entropy(out, y)
        l = cr_nll(out, d, c)
        return l + rank * deephit_rank(out, d, c) if rank else l
    for ep in range(epochs):
        model.train(); perm = rng.permutation(len(Xtr))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            l = loss_fn(Xtr[b], dtr[b], ctr[b])
            opt.zero_grad(); l.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(Xva, dva, cva).item()
        if v < best - 1e-4:
            best, state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8:
                break
    model.load_state_dict(state)
    return model


class Snapshot(nn.Module):
    """snapshot paradigm of the source study: current-state classifier of stress class within a window"""

    def __init__(self, d_in, D=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, D), nn.GELU(), nn.Dropout(0.1), nn.Linear(D, D), nn.GELU(),
                                 nn.Linear(D, K + 1))

    def forward(self, x):
        return self.net(x)
