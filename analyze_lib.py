"""Aggregation of saved predictions into metrics (point, alert/CRC, conformal lead-time bounds)."""
import glob
import os
import pickle
import numpy as np
import eval_lib as E

H = 21


def load_run(tier, split, model, seed, tag=""):
    f = f"results/{tier}_{split}_{model}_{seed}{tag}.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None


def load_tab(tier, split, seed):
    f = f"results/tab_{tier}_{split}_{seed}.pkl"
    return pickle.load(open(f, "rb")) if os.path.exists(f) else None


def shift_feats(pred, S, tier):
    cif = pred["cif"]
    tot = cif.sum(2)
    f = [tot[:, 6], tot[:, 13], tot[:, H - 1], pred["lm"] / 100.0]
    f += [cif[:, H - 1, j] for j in range(cif.shape[2])]
    return np.column_stack(f)


def full_eval(res, S, tier, alpha=0.1, alpha_m=0.1, window=None, a0=0.1, weighted=True):
    window = window or (14 if tier == "sim" else 7)
    cal, te = res["cal"], res["test"]
    out = E.point_metrics(te, window=window)
    tg = (0, 1, 2) if tier == "sim" else (0,)
    rc = cal["cif"][:, window - 1, list(tg)].sum(1); rt = te["cif"][:, window - 1, list(tg)].sum(1)
    th = E.crc_threshold(cal, S, rc, alpha_m, window=window, targets=tg)
    am, leads = E.alert_metrics(te, S, rt, th, cause_pred=te["cif"][:, H - 1].argmax(1), window=window, targets=tg)
    out.update({"alert_" + k: v for k, v in am.items()}); out["theta"] = th; out["leads"] = leads
    corr = E.lpb_calibrate(cal, alpha, a0)
    lpb = E.base_lower(te["cif"], a0) - corr
    out.update({"lpb_" + k: v for k, v in E.lpb_eval(lpb, te).items()})
    if weighted:
        fc, ft = shift_feats(cal, S, tier), shift_feats(te, S, tier)
        mu, sd = fc.mean(0), fc.std(0) + 1e-6
        lw, _ = E.lpb_weighted(cal, te, (fc - mu) / sd, (ft - mu) / sd, alpha, a0)
        out.update({"wlpb_" + k: v for k, v in E.lpb_eval(lw, te).items()})
    return out


def rule_eval(res, S, alpha_m=0.1, window=14):
    cal, te = res["cal"], res["test"]
    comp_c, comp_t = cal["comp"], te["comp"]
    rc, rt = comp_c.max(1), comp_t.max(1)
    th = E.crc_threshold(cal, S, rc, alpha_m, window=window)
    am, leads = E.alert_metrics(te, S, rt, th, cause_pred=comp_t.argmax(1), window=window)
    out = {"alert_" + k: v for k, v in am.items()}
    out["cindex_by_cause"] = [E.cindex_cr(comp_t[:, j], te["delta"], te["cause"], j) for j in range(3)]
    out["cindex"] = float(np.nanmean(out["cindex_by_cause"])); out["leads"] = leads
    from sklearn.metrics import f1_score
    ev = (te["cause"] >= 0) & (te["delta"] <= H)
    out["f1"] = float(f1_score(te["cause"][ev], comp_t[ev].argmax(1), average="macro"))
    return out
