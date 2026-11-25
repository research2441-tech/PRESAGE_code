"""Result figures (grayscale, no overlapping elements). Reads results/summary.json and saved predictions."""
import json
import pickle
import sys
import numpy as np
import figstyle  # noqa
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import presage_lib as PL
from analyze_lib import load_run, load_tab

S = json.load(open("results/summary.json"))
NAMES = {"presage": "PRESAGE", "hgb": "Gradient boosting", "deephit": "DeepHit-type MLP", "cox": "Cause-specific Cox",
         "snapshot": "Snapshot classifier", "transformer": "Time-aware Transformer", "grud": "GRU-D", "rules": "Index thresholds",
         "student": "PRESAGE student", "nopre": "w/o pretraining", "noanchor": "w/o anchoring", "nogap": "w/o elapsed time",
         "nosel": "w/o selectivity", "fed": "Federated", "local": "Local only"}
GRAYS = ["0.0", "0.35", "0.55", "0.7", "0.82"]
MARK = ["o", "s", "^", "D", "v", "P", "X"]


def save(fig, name):
    fig.savefig(f"figures/{name}.png", dpi=300)
    plt.close(fig)


def fig_forest():
    st = S["sim_temporal"]
    order = [m for m in ["presage", "hgb", "transformer", "grud", "deephit", "cox", "snapshot"] if m in st]
    fig, axs = plt.subplots(1, 3, figsize=(3.45, 2.55), sharey=True, gridspec_kw=dict(wspace=0.12))
    y = np.arange(len(order))[::-1]
    for ax, key, lab in zip(axs, ["cindex", "ibs", "f1"], ["C-index (↑)", "IBS (↓)", "Cause macro-F1 (↑)"]):
        for yi, mname in zip(y, order):
            v = st[mname].get(key, [np.nan, 0])
            if v[0] is None or np.isnan(v[0]):
                continue
            ax.errorbar(v[0], yi, xerr=v[1], fmt="o" if mname != "presage" else "s", color="black",
                        mfc="black" if mname == "presage" else "white", ms=3.6, lw=0.8, capsize=1.8)
        ax.set_xlabel(lab, fontsize=7)
        ax.xaxis.set_major_locator(mt.MaxNLocator(3))
        ax.tick_params(labelsize=6.5)
        ax.grid(axis="x", color="0.88", lw=0.5)
    axs[0].set_yticks(y); axs[0].set_yticklabels([NAMES[m] for m in order], fontsize=6.6)
    fig.subplots_adjust(left=0.34, right=0.98, bottom=0.17, top=0.97)
    save(fig, "fig3_main")


def fig_alerts():
    mc = S.get("sim_miss_curve", {})
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85), gridspec_kw=dict(wspace=0.45))
    ax = axs[0]
    for i, (mname, curve) in enumerate(mc.items()):
        a = [c["alpha"] for c in curve]; m = [c["miss"] for c in curve]
        ax.plot(a, m, marker=MARK[i], color=GRAYS[i], mfc="white" if i else "black", ms=3.2, lw=0.9, label=NAMES[mname])
    ax.plot([0, 0.32], [0, 0.32], ls=(0, (3, 2)), color="0.5", lw=0.7, label="nominal")
    ax.set_xlabel(r"target miss rate $\alpha_m$"); ax.set_ylabel("empirical miss rate")
    ax.set_xlim(0, 0.32); ax.set_ylim(0, 0.42)
    ax.legend(fontsize=5.6, loc="upper left", handlelength=1.6, borderaxespad=0.2)
    ax.text(-0.33, 1.02, "(a)", transform=ax.transAxes, fontsize=7.5)
    ax = axs[1]
    for i, (mname, curve) in enumerate(mc.items()):
        a = [c["alpha"] for c in curve]; f = [c["fa"] for c in curve]
        ax.plot(a, f, marker=MARK[i], color=GRAYS[i], mfc="white" if i else "black", ms=3.2, lw=0.9)
    ax.set_xlabel(r"target miss rate $\alpha_m$"); ax.set_ylabel("false-alert fraction")
    ax.set_xlim(0, 0.32); ax.set_ylim(0, 1.0)
    ax.text(-0.33, 1.02, "(b)", transform=ax.transAxes, fontsize=7.5)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.21, top=0.92)
    save(fig, "fig5_alerts")


def fig_leads(sims):
    """lead time before visible symptoms by cause at alpha_m = 0.1 (seed 0)"""
    from analyze_lib import full_eval
    import eval_lib as E
    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    pos, data, labels = [], [], []
    p0 = load_run("sim", "temporal", "presage", 0); t0 = load_tab("sim", "temporal", 0)
    for gi, (mname, res, comp) in enumerate([("presage", p0, False), ("hgb", t0["hgb"], False), ("rules", t0["rules"], True)]):
        cal, te = res["cal"], res["test"]
        if comp:
            rc, rt = cal["comp"].max(1), te["comp"].max(1)
        else:
            rc, rt = cal["cif"][:, 13].sum(1), te["cif"][:, 13].sum(1)
        th = E.crc_threshold(cal, sims, rc, 0.1)
        tab = E.season_alert_table(te, sims, rt, th)
        for j, cn in enumerate(["water", "nitrogen", "pathogen"]):
            leads = [sims[s]["vis"][j] - v["first"] for s, v in tab.items() if v["ev"] and v["hit"] and sims[s]["J"] == j
                     and sims[s]["vis"][j] is not None]
            data.append(leads); pos.append(j * 4 + gi)
    bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.0), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7))
    for i, b in enumerate(bp["boxes"]):
        b.set_facecolor(["0.25", "0.65", "white"][i // 3]); b.set_edgecolor("black"); b.set_linewidth(0.7)
    ax.set_xticks([1, 5, 9]); ax.set_xticklabels(["Water", "Nitrogen", "Pathogen"])
    ax.set_ylabel("days before visible symptoms")
    ax.axhline(0, color="0.6", lw=0.6, ls=(0, (2, 2)))
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=c, ec="black", lw=0.6, label=l) for c, l in
                       zip(["0.25", "0.65", "white"], ["PRESAGE", "Gradient boosting", "Index thresholds"])],
              fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, columnspacing=1.0, handlelength=1.2)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.14, top=0.84)
    save(fig, "fig6_leads")


def fig_robust():
    st = S["sim_temporal"]
    conds = ["clean", "rob_drop_thermal", "rob_drop_optical", "rob_drop_soil", "rob_drop_trap", "rob_weather_only"]
    lab = ["none", "thermal", "optical", "soil probe", "spore trap", "all but\nweather"]
    conds2 = ["clean", "rob_loss30", "rob_loss50", "rob_loss70"]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.05), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.32))
    models = [m for m in ["presage", "hgb", "transformer", "grud", "deephit"] if m in st]
    for i, mname in enumerate(models):
        r = st[mname]
        vals = [r["cindex"][0]] + [r["robust"].get(c, {}).get("cindex", [np.nan])[0] for c in conds[1:]]
        axs[0].plot(range(len(conds)), vals, marker=MARK[i], color=GRAYS[i], mfc="white" if i else "black", ms=3.0, lw=0.8,
                    label=NAMES[mname])
        vals2 = [r["cindex"][0]] + [r["robust"].get(c, {}).get("cindex", [np.nan])[0] for c in conds2[1:]]
        axs[1].plot([0, 30, 50, 70], vals2, marker=MARK[i], color=GRAYS[i], mfc="white" if i else "black", ms=3.0, lw=0.8)
    axs[0].set_xticks(range(len(conds))); axs[0].set_xticklabels(lab, rotation=35, ha="right", fontsize=6)
    axs[0].set_ylabel("C-index"); axs[0].set_xlabel("removed modality", fontsize=6.6)
    axs[1].set_xlabel("random token loss (%)", fontsize=6.6); axs[1].set_xticks([0, 30, 50, 70])
    lo = min(min(l.get_ydata().min() for l in a.get_lines()) for a in axs)
    for a in axs:
        a.set_ylim(np.floor(lo * 20) / 20 - 0.02, 0.9)
        a.grid(axis="y", color="0.9", lw=0.5)
    axs[0].legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.8, 1.27), ncol=3, columnspacing=0.8, handlelength=1.5)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.3, top=0.82)
    save(fig, "fig7_robust")


def fig_anchor(sims):
    from summarize import anchor_corr
    p0 = load_run("sim", "temporal", "presage", 0)
    corr, z, tr = anchor_corr(p0, sims)
    fig, axs = plt.subplots(1, 3, figsize=(3.45, 1.45), gridspec_kw=dict(wspace=0.55))
    labs = [("$z_1$ (CWSI channel)", "transpiration ratio"), ("$z_2$ (chlorophyll channel)", "N sufficiency"),
            ("$z_3$ (infection channel)", "log$_{10}$ infected area")]
    for c, ax in enumerate(axs):
        ax.hexbin(z[:, c], tr[:, c], gridsize=22, cmap="Greys", mincnt=1, bins="log", linewidths=0.1)
        ax.set_xlabel(labs[c][0], fontsize=5.8); ax.set_ylabel(labs[c][1], fontsize=5.8)
        ax.tick_params(labelsize=5.5)
        ax.set_title(r"$\rho_s$ = %.2f" % corr[c], fontsize=6.2, pad=2)
        ax.xaxis.set_major_locator(mt.MaxNLocator(3)); ax.yaxis.set_major_locator(mt.MaxNLocator(3))
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.27, top=0.86)
    save(fig, "fig8_anchor")


def fig_coverage():
    lz = S.get("lozo", {})
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85), gridspec_kw=dict(wspace=0.42))
    ax = axs[0]
    zn = ["coastal", "plain", "foothill", "alpine"]
    xs = np.arange(4); w = 0.36
    sc = [lz.get(str(z), {}).get("presage", {}).get("lpb_coverage_events", [np.nan])[0] for z in range(4)]
    wc = [lz.get(str(z), {}).get("presage", {}).get("wlpb_coverage_events", [np.nan])[0] for z in range(4)]
    ax.bar(xs - w / 2, sc, w, color="0.75", ec="black", lw=0.5, label="split conformal")
    ax.bar(xs + w / 2, wc, w, color="0.25", ec="black", lw=0.5, label="shift-weighted")
    ax.axhline(0.9, color="black", lw=0.7, ls=(0, (3, 2)))
    ax.set_xticks(xs); ax.set_xticklabels(zn, fontsize=6, rotation=25, ha="right")
    ax.set_ylabel("coverage of onsets"); ax.set_ylim(0.6, 1.02)
    ax.legend(fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2, columnspacing=0.8, handlelength=1.2)
    ax.text(-0.36, 1.06, "(a)", transform=ax.transAxes, fontsize=7.5)
    ax = axs[1]
    cc = S.get("sim_cov_curve", {})
    for i, (mname, curve) in enumerate(cc.items()):
        a = [1 - c["alpha"] for c in curve]
        ax.plot(a, [c["coverage_events"] for c in curve], marker=MARK[i], color=GRAYS[i], mfc="white" if i else "black",
                ms=3.2, lw=0.8, label=NAMES[mname])
    ax.plot([0.68, 0.97], [0.68, 0.97], color="0.5", lw=0.7, ls=(0, (3, 2)))
    ax.set_xlabel(r"nominal $1-\alpha$"); ax.set_ylabel("empirical coverage")
    ax.set_xlim(0.67, 0.97); ax.set_ylim(0.6, 1.0)
    ax.legend(fontsize=5.5, loc="upper left", handlelength=1.4, borderaxespad=0.2)
    ax.text(-0.36, 1.02, "(b)", transform=ax.transAxes, fontsize=7.5)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.25, top=0.83)
    save(fig, "fig9_coverage")


if __name__ == "__main__":
    sims = pickle.load(open("data/sim_fieldseasons.pkl", "rb"))
    for f in sys.argv[1:]:
        if f in ("leads", "anchor"):
            globals()["fig_" + f](sims)
        else:
            globals()["fig_" + f]()
    print("ok")
