"""Computes every number reported in the manuscript from saved predictions -> results/summary.json"""
import json
import os
import pickle
import sys
import numpy as np
from scipy import stats
import presage_lib as PL
import eval_lib as E
from analyze_lib import load_run, load_tab, full_eval, rule_eval, shift_feats

H = 21
OUT = "results/summary.json"


def mstd(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return [float(np.mean(v)), float(np.std(v)), len(v)] if v else [np.nan, np.nan, 0]


KEYS = ["cindex", "ibs", "f1", "ece", "alert_miss", "alert_lead_onset", "alert_lead_visible", "alert_false_alert_frac",
        "alert_fa_per_100lm", "alert_alert_rate", "alert_alert_cause_f1", "lpb_coverage", "lpb_coverage_events",
        "lpb_mean_lpb", "wlpb_coverage", "wlpb_coverage_events", "wlpb_mean_lpb"]


def season_ibs(pred):
    """per-season integrated Brier score (all causes), for paired tests"""
    cif, d, c, s = pred["cif"], pred["delta"], pred["cause"], pred["season"]
    hh = np.arange(1, H + 1)
    y = ((c[:, None, None] == np.arange(cif.shape[2])[None, None, :]) & (d[:, None, None] <= hh[None, :, None])).astype(float)
    keep = ~((c == -2) & (d < H))
    b = ((y - cif) ** 2).mean((1, 2))
    out = {}
    for sid in np.unique(s[keep]):
        out[int(sid)] = float(b[keep & (s == sid)].mean())
    return out


def collect(tier, split, S, models, tabmodels, seeds, tab_seeds, rob=False):
    res = {}
    perseason = {}
    for mname in models:
        rows, ps = [], []
        robs = {}
        for sd in seeds:
            r = load_run(tier, split, mname, sd)
            if r is None:
                continue
            o = full_eval(r, S, tier)
            o["time"] = r.get("time"); o["params"] = r.get("params"); o["latency"] = r.get("latency_ms_per_season")
            rows.append(o); ps.append(season_ibs(r["test"]))
            if rob:
                for k in [x for x in r if x.startswith("rob_")]:
                    pm = E.point_metrics(r[k], window=14 if tier == "sim" else 7)
                    robs.setdefault(k, []).append(pm)
        if rows:
            res[mname] = summarize_rows(rows)
            res[mname]["robust"] = {k: {kk: mstd([x[kk] for x in v]) for kk in ("cindex", "ibs", "f1")} for k, v in robs.items()}
            perseason[mname] = avg_dicts(ps)
    for mname in tabmodels:
        rows, ps, robs = [], [], {}
        for sd in tab_seeds:
            t = load_tab(tier, split, sd)
            if t is None or mname not in t:
                continue
            r = t[mname]
            if mname == "rules":
                o = rule_eval(r, S)
            else:
                o = full_eval(r, S, tier)
                ps.append(season_ibs(r["test"]))
            o["time"] = r.get("time")
            rows.append(o)
            if rob:
                for k in [x for x in r if x.startswith("rob_")]:
                    robs.setdefault(k, []).append(E.point_metrics(r[k], window=14 if tier == "sim" else 7))
        if rows:
            res[mname] = summarize_rows(rows)
            res[mname]["robust"] = {k: {kk: mstd([x[kk] for x in v]) for kk in ("cindex", "ibs", "f1")} for k, v in robs.items()}
            if ps:
                perseason[mname] = avg_dicts(ps)
    return res, perseason


def summarize_rows(rows):
    out = {k: mstd([r.get(k, np.nan) for r in rows]) for k in KEYS + ["time", "params", "latency", "theta"]}
    out["cindex_by_cause"] = [mstd([r["cindex_by_cause"][j] for r in rows]) for j in range(len(rows[0]["cindex_by_cause"]))]
    out["leads"] = [float(x) for x in rows[0].get("leads", [])]
    return out


def avg_dicts(ds):
    keys = set(ds[0]).intersection(*map(set, ds[1:])) if len(ds) > 1 else set(ds[0])
    return {k: float(np.mean([d[k] for d in ds])) for k in keys}


def paired_tests(perseason, ref="presage"):
    out = {}
    if ref not in perseason:
        return out
    pv = []
    for mname, d in perseason.items():
        if mname == ref:
            continue
        ks = sorted(set(d) & set(perseason[ref]))
        a = np.array([perseason[ref][k] for k in ks]); b = np.array([d[k] for k in ks])
        st = stats.wilcoxon(a, b)
        diff = b - a
        rbc = (np.sum(diff > 0) - np.sum(diff < 0)) / max(np.sum(diff != 0), 1)  # matched-pairs rank-biserial (sign)
        out[mname] = dict(n=len(ks), ref_mean=float(a.mean()), other_mean=float(b.mean()), p=float(st.pvalue),
                          median_diff=float(np.median(diff)), effect=float(rbc))
        pv.append((mname, st.pvalue))
    # Holm correction
    pv.sort(key=lambda x: x[1]); m = len(pv); prev = 0
    for i, (k, p) in enumerate(pv):
        adj = min(1.0, max(prev, (m - i) * p)); prev = adj
        out[k]["p_holm"] = float(adj)
    # Friedman across all models with complete seasons
    names = list(perseason)
    if len(names) < 3:
        return out
    ks = sorted(set.intersection(*[set(perseason[n]) for n in names]))
    mat = np.array([[perseason[n][k] for n in names] for k in ks])
    fr = stats.friedmanchisquare(*mat.T)
    ranks = stats.rankdata(mat, axis=1).mean(0)
    out["_friedman"] = dict(chi2=float(fr.statistic), p=float(fr.pvalue), n=len(ks),
                            mean_ranks=dict(zip(names, map(float, ranks))),
                            cd=float(nemenyi_cd(len(names), len(ks))))
    return out


def nemenyi_cd(k, n, alpha=0.05):
    q = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268}[k]
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def bootstrap_dc(S, a, b, n_boot=200, seed=0):
    """paired season bootstrap of the difference in mean cause-specific C-index (b - a) using seed-0 predictions"""
    rng = np.random.default_rng(seed)
    sids = np.unique(a["season"]); diffs = []
    idx_a = {s: np.where(a["season"] == s)[0] for s in sids}; idx_b = {s: np.where(b["season"] == s)[0] for s in sids}
    for _ in range(n_boot):
        pick = rng.choice(sids, len(sids))
        ia = np.concatenate([idx_a[s] for s in pick]); ib = np.concatenate([idx_b[s] for s in pick])
        ca = np.nanmean([E.cindex_cr(a["cif"][ia, H - 1, j], a["delta"][ia], a["cause"][ia], j) for j in range(a["cif"].shape[2])])
        cb = np.nanmean([E.cindex_cr(b["cif"][ib, H - 1, j], b["delta"][ib], b["cause"][ib], j) for j in range(b["cif"].shape[2])])
        diffs.append(cb - ca)
    return [float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def miss_curve(res, S, tier, alphas=(0.05, 0.1, 0.15, 0.2, 0.3), window=14, comp=False):
    cal, te = res["cal"], res["test"]
    out = []
    for am in alphas:
        if comp:
            rc, rt, cp = cal["comp"].max(1), te["comp"].max(1), te["comp"].argmax(1)
        else:
            rc = cal["cif"][:, window - 1].sum(1); rt = te["cif"][:, window - 1].sum(1); cp = te["cif"][:, H - 1].argmax(1)
        th = E.crc_threshold(cal, S, rc, am, window=window)
        m, leads = E.alert_metrics(te, S, rt, th, cause_pred=cp, window=window)
        out.append(dict(alpha=am, miss=m["miss"], fa=m["false_alert_frac"], lead=m["lead_visible"], rate=m["alert_rate"]))
    return out


def coverage_curve(res, alphas=(0.05, 0.1, 0.2, 0.3)):
    out = []
    for a in alphas:
        c = E.lpb_calibrate(res["cal"], a)
        lpb = E.base_lower(res["test"]["cif"]) - c
        ev = E.lpb_eval(lpb, res["test"])
        out.append(dict(alpha=a, coverage=ev["coverage"], coverage_events=ev["coverage_events"], mean=ev["mean_lpb"]))
    return out


def anchor_corr(res, S):
    te = res["test"]; z = te["z"]
    tr = np.array([[S[s]["truth"]["rftra"][int(l)], S[s]["truth"]["ns"][int(l)],
                    np.log10(S[s]["truth"]["L"][int(l)] + S[s]["truth"]["S"][int(l)] + 1e-5)] for s, l in zip(te["season"], te["lm"])])
    return [float(stats.spearmanr(z[:, c], tr[:, c]).statistic) for c in range(3)], z, tr


def main():
    summ = {}
    if os.path.exists(OUT):
        summ = json.load(open(OUT))
    which = sys.argv[1:] or ["sim", "real", "lozo"]
    if "sim" in which:
        S = pickle.load(open("data/sim_fieldseasons.pkl", "rb"))
        models = ["presage", "student", "transformer", "grud", "nopre", "noanchor", "nogap", "nosel", "fed", "local"]
        res, ps = collect("sim", "temporal", S, models, ["hgb", "deephit", "snapshot", "cox", "rules"], range(5), range(5), rob=True)
        summ["sim_temporal"] = res
        main_set = {k: v for k, v in ps.items() if k in ("presage", "transformer", "grud", "hgb", "deephit", "snapshot", "cox")}
        summ["sim_tests"] = paired_tests(main_set)
        abl_set = {k: v for k, v in ps.items() if k in ("presage", "nopre", "noanchor", "nogap", "nosel")}
        summ["sim_abl_tests"] = paired_tests(abl_set)
        p0 = load_run("sim", "temporal", "presage", 0)
        if p0 is not None:
            t0 = load_tab("sim", "temporal", 0)
            summ["sim_dc"] = {}
            for name, other in [("hgb", t0["hgb"]["test"] if t0 else None), ("transformer", (load_run("sim", "temporal", "transformer", 0) or {}).get("test")),
                                ("grud", (load_run("sim", "temporal", "grud", 0) or {}).get("test"))]:
                if other is not None:
                    summ["sim_dc"][name] = bootstrap_dc(S, p0["test"], other)
            summ["sim_miss_curve"] = {"presage": miss_curve(p0, S, "sim")}
            if t0:
                summ["sim_miss_curve"]["hgb"] = miss_curve(t0["hgb"], S, "sim")
                summ["sim_miss_curve"]["rules"] = miss_curve(t0["rules"], S, "sim", comp=True)
            summ["sim_cov_curve"] = {"presage": coverage_curve(p0)}
            if t0:
                summ["sim_cov_curve"]["hgb"] = coverage_curve(t0["hgb"])
            corr, _, _ = anchor_corr(p0, S)
            summ["anchor_corr"] = corr
    if "sens" in which:
        out = {}
        for tag, fn in [("_k0", "data/sim_fieldseasons_k0.pkl"), ("", "data/sim_fieldseasons.pkl"), ("_k2", "data/sim_fieldseasons_k2.pkl")]:
            if not os.path.exists(fn):
                continue
            Sk = pickle.load(open(fn, "rb"))
            row = {}
            p = load_run("sim", "temporal", "presage", 0, tag)
            t = load_tab("sim", "temporal", 0) if tag == "" else (pickle.load(open(f"results/tab_sim_temporal_0{tag}.pkl", "rb"))
                                                                if os.path.exists(f"results/tab_sim_temporal_0{tag}.pkl") else None)
            for name, res in [("presage", p), ("hgb", t["hgb"] if t else None)]:
                if res is None:
                    continue
                o = full_eval(res, Sk, "sim", weighted=False)
                # pathogen-only lead before visible symptoms
                import eval_lib as E2
                rc = res["cal"]["cif"][:, 13].sum(1); rt = res["test"]["cif"][:, 13].sum(1)
                th = E2.crc_threshold(res["cal"], Sk, rc, 0.1)
                tab = E2.season_alert_table(res["test"], Sk, rt, th)
                lp = [Sk[s_]["vis"][2] - v["first"] for s_, v in tab.items() if v["ev"] and v["hit"] and Sk[s_]["J"] == 2
                      and Sk[s_]["vis"][2] is not None]
                row[name] = dict(cindex=o["cindex"], cP=o["cindex_by_cause"][2], miss=o["alert_miss"],
                                 lead=o["alert_lead_visible"], leadP=float(np.median(lp)) if lp else float("nan"))
            out[{"_k0": "0", "": "1", "_k2": "2"}[tag]] = row
        summ["sens_kappa"] = out
        S = pickle.load(open("data/sim_fieldseasons.pkl", "rb"))
        p0 = load_run("sim", "temporal", "presage", 0); t0 = load_tab("sim", "temporal", 0)
        ws = {}
        for name, res in [("presage", p0), ("hgb", t0["hgb"])]:
            ws[name] = []
            for w in (7, 14, 21):
                o = full_eval(res, S, "sim", window=w, weighted=False)
                ws[name].append(dict(W=w, miss=o["alert_miss"], fa=o["alert_false_alert_frac"], lead=o["alert_lead_visible"],
                                     leadon=o["alert_lead_onset"]))
        summ["sens_window"] = ws
    if "lozo" in which:
        S = pickle.load(open("data/sim_fieldseasons.pkl", "rb"))
        summ["lozo"] = {}
        for z in range(4):
            res, _ = collect("sim", f"lozo{z}", S, ["presage"], ["hgb", "deephit", "cox"], [0], [0])
            summ["lozo"][str(z)] = res
    if "real" in which:
        S = pickle.load(open("data/real_episodes.pkl", "rb"))
        PL.K = 2
        summ["real"] = {}
        for f in range(3):
            res, ps = collect("real", f"loco{f}", S, ["presage", "nopre", "noanchor", "transformer", "grud", "fed", "nogap"],
                              ["hgb", "deephit", "snapshot", "cox"], range(3), range(3), rob=True)
            summ["real"][str(f)] = res
    json.dump(summ, open(OUT, "w"), indent=1, default=float)
    print("saved", OUT)


if __name__ == "__main__":
    main()
