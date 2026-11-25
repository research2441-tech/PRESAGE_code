"""Tabular, rule and snapshot baselines for one tier x split x seed."""
import argparse
import os
import pickle
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import presage_lib as PL
from presage_lib import Prep, season_arrays, cif_from_logits
import baselines_tab as BT
from run_exp import get_data, get_split, ROB

RFEATS = ["depth", "sand", "silt", "clay", "rzd", "soilflag", "day", "tmax7", "rh7", "rain7", "rain30", "et07", "def7",
          "sm07", "ten_last", "ten_d2", "ten_d5", "ten_max10", "ten_age", "ndvi_last", "ndmi_last", "ndmi_d", "rs_age",
          "irr14", "irr_age"]


def real_features(s, drop_mod=None, token_drop=0.0, rng=None):
    t, m, v = season_arrays(s)
    keep = np.ones(len(t), bool)
    if drop_mod is not None:
        for dm in np.atleast_1d(drop_mod):
            keep &= m != dm
    if token_drop > 0:
        keep &= ~((rng.random(len(t)) < token_drop) & (m != 0))
    t, m, v = t[keep], m[keep], v[keep]
    rows = []
    for lm in s["landmarks"]:
        c = t < lm + 1.0
        f = dict(zip(RFEATS[:6], v[m == 0][0][:6])); f["day"] = lm
        w = c & (m == 1); wl = v[w]; wt = t[w]; r7 = wt >= lm - 6; r30 = wt >= lm - 29
        f["tmax7"] = wl[r7, 1].mean() if r7.any() else np.nan
        f["rh7"] = wl[r7, 2].mean() if r7.any() else np.nan
        f["rain7"] = np.expm1(wl[r7, 3]).sum() if r7.any() else np.nan
        f["rain30"] = np.expm1(wl[r30, 3]).sum() if r30.any() else np.nan
        f["et07"] = wl[r7, 6].mean() if r7.any() else np.nan
        f["def7"] = (wl[r7, 6] - np.expm1(wl[r7, 3])).mean() if r7.any() else np.nan
        f["sm07"] = wl[-1, 7] if len(wl) else np.nan
        so = c & (m == 2); sv = v[so, 0]; st = t[so]
        f["ten_last"] = sv[-1] if len(sv) else np.nan
        f["ten_d2"] = sv[-1] - sv[st <= st[-1] - 2][-1] if len(sv) and (st <= st[-1] - 2).any() else np.nan
        f["ten_d5"] = sv[-1] - sv[st <= st[-1] - 5][-1] if len(sv) and (st <= st[-1] - 5).any() else np.nan
        f["ten_max10"] = sv[st >= lm - 9].max() if len(sv) and (st >= lm - 9).any() else np.nan
        f["ten_age"] = lm + 1 - st[-1] if len(st) else 99
        op = c & (m == 3); ov = v[op]; ot = t[op]
        f["ndvi_last"] = ov[-1, 0] if len(ov) else np.nan
        f["ndmi_last"] = ov[-1, 1] if len(ov) else np.nan
        f["ndmi_d"] = ov[-1, 1] - ov[-2, 1] if len(ov) > 1 else np.nan
        f["rs_age"] = lm + 1 - ot[-1] if len(ot) else 99
        mg = c & (m == 5); mv = v[mg]; mt = t[mg]
        f["irr14"] = mv[mt >= lm - 13, 1].sum() if len(mv) else 0.0
        f["irr_age"] = lm + 1 - mt[-1] if len(mt) else 99
        rows.append([f[k] for k in RFEATS])
    return np.array(rows, np.float32)


def table(tier, idx, S, prep, **kw):
    if tier == "sim":
        if "token_drop" in kw:
            rng = np.random.default_rng(99)
            tabs = []
            X, Sx, LM, Dl, C = [], [], [], [], []
            for i in idx:
                s = S[i]
                if not s["landmarks"]:
                    continue
                t, m, v = season_arrays(s)
                keep = ~((rng.random(len(t)) < kw["token_drop"]) & (m != 0))
                X.append(BT.landmark_features(s, prep, keep_mask=keep))
                Sx.append(np.full(len(s["landmarks"]), i)); LM.append(np.array(s["landmarks"]))
                lms, d, c = PL.landmark_targets(s); Dl.append(d); C.append(c)
            return dict(X=np.concatenate(X), season=np.concatenate(Sx), lm=np.concatenate(LM).astype(float),
                        delta=np.concatenate(Dl), cause=np.concatenate(C))
        return BT.landmark_table(idx, S, prep, **kw)
    rng = np.random.default_rng(99)
    X, Sx, LM, Dl, C = [], [], [], [], []
    for i in idx:
        s = S[i]
        X.append(real_features(s, rng=rng, **kw)); Sx.append(np.full(len(s["landmarks"]), i))
        LM.append(np.array(s["landmarks"])); lms, d, c = PL.landmark_targets(s); Dl.append(d); C.append(c)
    return dict(X=np.concatenate(X), season=np.concatenate(Sx), lm=np.concatenate(LM).astype(float),
                delta=np.concatenate(Dl), cause=np.concatenate(C))


def pack(tab, cif):
    return dict(season=tab["season"], lm=tab["lm"], delta=tab["delta"], cause=tab["cause"], cif=cif)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="sim"); ap.add_argument("--split", default="temporal")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--rob", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    S = get_data(a.tier)
    BT.K = PL.K
    sp = get_split(a.tier, a.split, S, a.seed)
    prep = Prep().fit([S[i] for i in sp["train"]])
    T = {k: table(a.tier, v, S, prep) for k, v in sp.items()}
    if a.tier == "real":
        BT.FEATS = RFEATS
    imp = BT.Imputer().fit(T["train"]["X"])
    out = {}
    t0 = time.time()
    # discrete-time gradient boosting
    clf = BT.hgb_fit(T["train"], seed=a.seed)
    out["hgb"] = dict(cal=pack(T["cal"], BT.hgb_cif(clf, T["cal"])), test=pack(T["test"], BT.hgb_cif(clf, T["test"])),
                      time=time.time() - t0)
    # DeepHit-type MLP
    t0 = time.time()
    torch.manual_seed(a.seed)
    mlp = BT.MLPCR(imp(T["train"]["X"]).shape[1])
    mlp = BT.train_mlp(mlp, imp(T["train"]["X"]), T["train"]["delta"], T["train"]["cause"], imp(T["val"]["X"]),
                       T["val"]["delta"], T["val"]["cause"], seed=a.seed)
    mlp.eval()
    f = lambda tb: cif_from_logits(mlp(torch.tensor(imp(tb["X"])))).detach().numpy()
    out["deephit"] = dict(cal=pack(T["cal"], f(T["cal"])), test=pack(T["test"], f(T["test"])), time=time.time() - t0)
    # snapshot classifier (current-state stress class within the alert window); expressed as a flat CIF
    t0 = time.time()
    torch.manual_seed(a.seed)
    sn = BT.Snapshot(imp(T["train"]["X"]).shape[1], D=64)
    sn.net[-1] = torch.nn.Linear(64, PL.K + 1)
    sn = BT.train_mlp(sn, imp(T["train"]["X"]), T["train"]["delta"], T["train"]["cause"], imp(T["val"]["X"]),
                      T["val"]["delta"], T["val"]["cause"], seed=a.seed, snapshot=True)
    sn.eval()

    def fsn(tb):
        p = torch.softmax(sn(torch.tensor(imp(tb["X"]))), -1).detach().numpy()[:, 1:]
        ramp = np.clip(np.arange(1, PL.H + 1) / 14.0, 0, 1)  # uniform-in-time spreading inside the window
        return p[:, None, :] * ramp[None, :, None]
    out["snapshot"] = dict(cal=pack(T["cal"], fsn(T["cal"])), test=pack(T["test"], fsn(T["test"])), time=time.time() - t0)
    if a.seed == 0:
        # cause-specific Cox (deterministic)
        t0 = time.time()
        try:
            cox = BT.cox_fit(T["train"], imp)
            out["cox"] = dict(cal=pack(T["cal"], BT.cox_cif(cox, T["cal"], imp)), test=pack(T["test"], BT.cox_cif(cox, T["test"], imp)),
                              time=time.time() - t0)
        except Exception as e:
            print("cox failed", e)
        if a.tier == "sim":
            out["rules"] = dict(cal=dict(**pack(T["cal"], None), comp=BT.rule_scores(T["cal"])),
                                test=dict(**pack(T["test"], None), comp=BT.rule_scores(T["test"])))
    if a.rob:
        for k, kw in ROB.items():
            if a.tier == "real" and k in ("drop_thermal", "drop_trap"):
                continue
            tb = table(a.tier, sp["test"], S, prep, **kw)
            out["hgb"]["rob_" + k] = pack(tb, BT.hgb_cif(clf, tb))
            out["deephit"]["rob_" + k] = pack(tb, f(tb))
    os.makedirs("results", exist_ok=True)
    pickle.dump(out, open(f"results/tab_{a.tier}_{a.split}_{a.seed}{a.tag}.pkl", "wb"))
    print("done", a.tier, a.split, a.seed)


if __name__ == "__main__":
    main()
