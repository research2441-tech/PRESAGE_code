"""Fig. 2 (case study of two test seasons) and Fig. 4 (learned memory and selectivity)."""
import pickle
import sys
import numpy as np
import torch
import figstyle  # noqa
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import presage_lib as PL
import eval_lib as E
from analyze_lib import load_run
from train import batches
from splits import temporal_split

S = pickle.load(open("data/sim_fieldseasons.pkl", "rb"))
MODN = ["static", "weather", "soil probe", "optical", "UAV thermal", "management", "spore trap"]


def pick(res, cause):
    te = res["test"]
    best, bs = None, -1
    for s in np.unique(te["season"]):
        sv = S[s]
        if sv["J"] != cause or sv["vis"][cause] is None or sv["T"] < 40:
            continue
        sel = te["season"] == s
        r = te["cif"][sel, 13].sum(1)
        lead = sv["vis"][cause] - sv["T"]
        score = lead + 10 * r[-1] - 5 * r[0]
        if score > bs:
            best, bs = s, score
    return best


def fig_case():
    res = load_run("sim", "temporal", "presage", 0)
    cal = res["cal"]; te = res["test"]
    th = E.crc_threshold(cal, S, cal["cif"][:, 13].sum(1), 0.1)
    seasons = [pick(res, 0), pick(res, 2)]
    fig, axs = plt.subplots(3, 2, figsize=(7.0, 4.05), sharex="col",
                            gridspec_kw=dict(height_ratios=[1.15, 1.0, 1.1], hspace=0.16, wspace=0.08))
    for col, s in enumerate(seasons):
        sv = S[s]; T = sv["T"]; J = sv["J"]; V = sv["vis"][J]
        ax = axs[0, col]
        for m in range(1, 7):
            t = sv["tok_t"][sv["tok_m"] == m]
            ax.plot(t, np.full(len(t), 7 - m), "|", color="black", ms=4.5 if m != 1 else 3.0, mew=0.7)
        ax.set_yticks(range(1, 7)); ax.set_yticklabels(MODN[1:][::-1] if col == 0 else [], fontsize=6)
        ax.set_ylim(0.3, 7.5)
        ax.set_title(["Season A: water stress first", "Season B: pathogen infection first"][col], fontsize=7.4, pad=3)
        ax = axs[1, col]
        d = np.arange(sv["n"]); emerged = sv["truth"]["lai"] >= 0.05
        mask = lambda a: np.where(emerged, a, np.nan)
        ax.plot(d, mask(sv["truth"]["rftra"]), color="black", lw=0.9, label="transpiration ratio")
        ax.plot(d, mask(np.clip(sv["truth"]["ns"], 0, 1.2)), color="0.45", lw=0.9, ls=(0, (4, 1.5)), label="N sufficiency")
        inf = np.clip((sv["truth"]["L"] + sv["truth"]["S"]) / 0.02, 0, 1.2)
        ax.plot(d, mask(inf), color="0.6", lw=0.9, ls=(0, (1, 1)), label="infected area (scaled)")
        ax.set_ylim(-0.08, 1.3); ax.set_yticks([0, 0.5, 1.0]); ax.set_ylabel("simulated state" if col == 0 else "", fontsize=6.6)

        ax = axs[2, col]
        sel = te["season"] == s
        lm = te["lm"][sel]; cif = te["cif"][sel]
        styles = [("water", "black", "-"), ("nitrogen", "0.45", (0, (4, 1.5))), ("pathogen", "0.6", (0, (1, 1)))]
        for j, (nm, c, ls) in enumerate(styles):
            ax.plot(lm, cif[:, 13, j], color=c, ls=ls, lw=1.0, marker="o", ms=2.0, label=f"$F_{{{nm[0].upper()}}}(14)$")
        r = cif[:, 13].sum(1)
        ax.axhline(th, color="black", lw=0.6, ls=(0, (6, 2)))
        al = lm[r >= th]
        if len(al):
            ax.plot(al, np.full(len(al), 1.03), "v", color="black", ms=3.2, clip_on=False)
        ax.set_ylim(0, 1.08); ax.set_ylabel("cumulative incidence" if col == 0 else "", fontsize=6.6)
        ax.set_xlabel("days after sowing", fontsize=6.8)
        if col == 0:
            ax.text(lm[0] - 4, th + 0.03, r"$\hat{\theta}$", fontsize=6.5)
        for a in axs[:, col]:
            a.axvline(T, color="black", lw=0.8)
            if V is not None:
                a.axvline(V, color="0.5", lw=0.8, ls=(0, (2, 1.5)))
            a.tick_params(labelsize=6)
        axs[0, col].text(T - 0.8, 6.95, "onset", fontsize=5.8, ha="right", va="center")
        if V is not None:
            axs[0, col].text(V + 0.8, 6.95, "visible", fontsize=5.8, ha="left", va="center")
        axs[2, col].set_xlim(0, max(V or T, T) + 12)
    h1, l1 = axs[1, 0].get_legend_handles_labels(); h2, l2 = axs[2, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    extra = [Line2D([], [], color="black", lw=0.8), Line2D([], [], color="0.5", lw=0.8, ls=(0, (2, 1.5))),
             Line2D([], [], color="black", ls="none", marker="v", ms=3.2)]
    fig.legend(h1 + h2 + extra, l1 + l2 + ["onset", "visible symptoms", "alert"], loc="lower center", ncol=9,
               fontsize=5.8, handlelength=1.8, columnspacing=1.0, frameon=False, bbox_to_anchor=(0.54, 0.0))
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.135, top=0.93)
    fig.savefig("figures/fig2_case.png", dpi=300)
    plt.close(fig)
    return seasons, th


def fig_mech():
    sp = temporal_split(S)
    prep = PL.Prep().fit([S[i] for i in sp["train"]])
    enc = [prep.encode(s) for s in S]
    m = PL.PRESAGE(D=64)
    m.load_state_dict(torch.load("models/sim_temporal_presage_0.pt")); m.eval()
    taus = [1.0 / torch.exp(l.log_lam).detach().numpy().ravel() for l in m.enc.layers]
    tau0 = np.exp(np.linspace(np.log(0.7), np.log(60.0), 8))
    caps = {}

    def hook(layer, idx):
        def f(mod, inp, out):
            caps.setdefault(idx, []).append(out.detach())
        return f
    hs = [l.dlt.register_forward_hook(hook(l, i)) for i, l in enumerate(m.enc.layers)]
    deltas = {k: [] for k in range(1, 7)}
    with torch.no_grad():
        for bidx, b in batches(sp["test"][:120], enc, S, 32, np.random.default_rng(0), shuffle=False):
            caps.clear(); m.enc(b)
            dl = torch.nn.functional.softplus(caps[0][0]).mean(-1)  # mean step over channels, layer 1
            for k in range(1, 7):
                sel = (b["m"] == k) & ~b["pad"]
                deltas[k] += dl[sel].tolist()
    for h in hs:
        h.remove()
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.85), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1.1]))
    ax = axs[0]
    init = np.tile(tau0, taus[0].size // tau0.size)
    for li, (tt, col, mk) in enumerate([(taus[0], "black", "o"), (taus[1], "0.55", "s")]):
        ratio = tt / init
        ax.scatter(init * np.exp(np.random.default_rng(li).normal(0, 0.04, init.size)), ratio, s=2, color=col,
                   marker=mk, lw=0, alpha=0.6, label=f"layer {li + 1}")
    ax.axhline(1.0, color="0.4", lw=0.6, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xlabel(r"initial time constant (days)", fontsize=6.4); ax.set_ylabel(r"learned / initial $1/\lambda$", fontsize=6.4)
    ax.legend(fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.0, markerscale=2.5,
              columnspacing=0.8, borderaxespad=0.1)
    ax.text(-0.42, 1.02, "(a)", transform=ax.transAxes, fontsize=7.5)
    ax.tick_params(labelsize=6)
    ax = axs[1]
    data = [np.array(deltas[k]) for k in range(1, 7)]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6, medianprops=dict(color="black", lw=0.9),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for b_ in bp["boxes"]:
        b_.set_facecolor("0.8"); b_.set_edgecolor("black"); b_.set_linewidth(0.6)
    ax.set_xticks(range(1, 7)); ax.set_xticklabels(["weather", "soil", "optical", "thermal", "mgmt", "trap"], rotation=35,
                                                   ha="right", fontsize=6)
    ax.set_ylabel(r"mean step $\Delta_k$ (days)", fontsize=6.4)
    ax.text(-0.3, 1.02, "(b)", transform=ax.transAxes, fontsize=7.5)
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.26, top=0.86)
    fig.savefig("figures/fig4_mechanism.png", dpi=300)
    plt.close(fig)
    return {k: float(np.median(v)) for k, v in deltas.items()}, [float(np.median(t)) for t in taus], \
        [float(np.mean(t > 60)) for t in taus]


if __name__ == "__main__":
    import json
    out = {}
    if "case" in sys.argv:
        s, th = fig_case(); out["case"] = dict(seasons=[int(x) for x in s], theta=th)
    if "mech" in sys.argv:
        d, t, f = fig_mech(); out["mech"] = dict(delta_median=d, tau_median=t, frac_tau_gt60=f)
    json.dump(out, open("results/fig_case_mech.json", "w"), indent=1)
    print(out)
