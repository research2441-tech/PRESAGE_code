"""Fig. 10 (real consortia) and Fig. 11 (accuracy versus model size)."""
import json
import sys
import numpy as np
import figstyle  # noqa
import matplotlib.pyplot as plt

S = json.load(open("results/summary.json"))
NAMES = {"presage": "PRESAGE", "hgb": "Gradient boosting", "deephit": "DeepHit-type MLP", "cox": "Cox",
         "transformer": "Transformer", "grud": "GRU-D", "nopre": "PRESAGE w/o pretraining", "student": "Student",
         "snapshot": "Snapshot"}
FILLS = ["0.0", "0.3", "0.5", "0.7", "0.85", "white"]
HATCH = ["", "", "", "", "", "////"]


def fig_real():
    rr = S["real"]
    models = [k for k in ["presage", "grud", "transformer", "hgb", "deephit", "cox"] if any(k in rr.get(str(f), {}) for f in range(3))]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.95), gridspec_kw=dict(width_ratios=[1.6, 0.8], wspace=0.75))
    ax = axs[0]
    w = 0.8 / len(models); xs = np.arange(3)
    for i, k in enumerate(models):
        mu = [rr.get(str(f), {}).get(k, {}).get("cindex", [np.nan, 0])[0] for f in range(3)]
        sd = [rr.get(str(f), {}).get(k, {}).get("cindex", [np.nan, 0])[1] for f in range(3)]
        ax.bar(xs - 0.4 + w / 2 + i * w, mu, w, yerr=sd, color=FILLS[i], ec="black", lw=0.4, hatch=HATCH[i],
               error_kw=dict(lw=0.5, capsize=1.2), label=NAMES[k])
    ax.axhline(0.5, color="0.5", lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(xs); ax.set_xticklabels(["C0 held out", "C1 held out", "C2 held out"], fontsize=6.2)
    ax.set_ylabel("C-index"); ax.set_ylim(0.4, 0.9)
    ax.legend(fontsize=5.4, ncol=3, loc="lower center", bbox_to_anchor=(0.75, 1.0), columnspacing=0.7, handlelength=1.1)
    ax.text(-0.30, 1.04, "(a)", transform=ax.transAxes, fontsize=7.5)
    ax = axs[1]
    for i, k in enumerate(models):
        mi = [rr.get(str(f), {}).get(k, {}).get("alert_miss", [np.nan])[0] for f in range(3)]
        ax.barh(len(models) - 1 - i, np.nanmean(mi), 0.7, color=FILLS[i], ec="black", lw=0.4, hatch=HATCH[i])
    ax.axvline(0.1, color="0.4", lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks(range(len(models))); ax.set_yticklabels([NAMES[k] for k in models][::-1], fontsize=5.6)
    ax.set_xlabel("miss rate ($\\alpha_m$ = 0.1)", fontsize=6.4); ax.set_xlim(0, 0.55)
    ax.text(-0.75, 1.02, "(b)", transform=ax.transAxes, fontsize=7.5)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.17, top=0.8)
    fig.savefig("figures/fig10_real.png", dpi=300); plt.close(fig)


def fig_eff():
    st = S["sim_temporal"]
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    pts = [k for k in ["presage", "student", "transformer", "grud"] if k in st]
    offs = {"presage": (8, 0), "student": (-7, 0), "transformer": (9, -6), "grud": (8, 6)}
    has = {"presage": "left", "student": "right", "transformer": "left", "grud": "left"}
    for i, k in enumerate(pts):
        x = st[k]["latency"][0]; y = st[k]["cindex"][0]
        ax.scatter(x, y, s=12 + st[k]["params"][0] / 1500, color=FILLS[i], ec="black", lw=0.6, zorder=3)
        ax.annotate(f"{NAMES[k]} ({st[k]['params'][0]/1e3:.0f}k)", (x, y), xytext=offs[k], textcoords="offset points",
                    fontsize=5.8, va="center", ha=has[k])
    ax.set_xlabel("CPU latency per season (ms, single thread)"); ax.set_ylabel("C-index")
    lo = min(st[k]["cindex"][0] for k in pts); hi = max(st[k]["cindex"][0] for k in pts)
    ax.set_ylim(lo - 0.008, hi + 0.008); ax.set_xlim(0, 14)
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.21, top=0.95)
    fig.savefig("figures/fig11_eff.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    for a in sys.argv[1:]:
        globals()["fig_" + a]()
    print("ok")
