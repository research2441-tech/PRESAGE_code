"""Fig. 1: PRESAGE architecture (grayscale). Forward data flow = solid arrows; reverse-mode gradient
paths from the training objectives = dashed arrows."""
import figstyle  # noqa
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig = plt.figure(figsize=(7.16, 4.05), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56.5); ax.axis("off")

BOX = {}


def box(key, x, y, w, h, text, fill="white", lw=0.8, style="round,pad=0.02,rounding_size=0.8", fs=7.0, bold=False, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fill, ec="black", lw=lw, ls=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.15)
    BOX[key] = (x, y, w, h)


def side(key, s, f=0.5):
    x, y, w, h = BOX[key]
    return {"l": (x, y + f * h), "r": (x + w, y + f * h), "t": (x + f * w, y + h), "b": (x + f * w, y)}[s]


def arrow(p0, p1, dashed=False, lw=0.8, rad=0.0, head=6):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=head, lw=lw, color="black",
                        ls=(0, (3.0, 2.0)) if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def poly(pts, dashed=False, lw=0.8, head=6):
    """orthogonal polyline with an arrow head on the last segment"""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color="black", lw=lw, ls=(0, (3.0, 2.0)) if dashed else "-", solid_capstyle="butt")
    arrow(pts[-2], pts[-1], dashed=dashed, lw=lw, head=head)


# ---------------- input modalities (left column)
mods = ["Weather station (daily)", "Soil-moisture probe (outages)", "Red-edge / NDVI (5-d, clouds)",
        "UAV thermal (irregular)", "Spore trap (weekly)", "Management log (events)", "Static field descriptors"]
ytop = 51.0; hm = 4.6; gap = 1.75
ax.text(10.3, 54.4, "Asynchronous field streams", ha="center", va="center", fontsize=7.4, fontweight="bold")
for i, m in enumerate(mods):
    box(f"m{i}", 1.0, ytop - i * (hm + gap) - hm, 18.6, hm, m, fill="0.93", fs=6.6)
# brace collector line
xc = 21.2
ys = [side(f"m{i}", "r")[1] for i in range(len(mods))]
for i in range(len(mods)):
    ax.plot([side(f"m{i}", "r")[0], xc], [ys[i], ys[i]], color="black", lw=0.7)
ax.plot([xc, xc], [ys[-1], ys[0]], color="black", lw=0.7)

# ---------------- tokenizer and encoder
box("tok", 23.5, 30.0, 14.2, 13.0, "Event tokenizer\n$(t_k,\\,m_k,\\,\\mathbf{x}_k)\\rightarrow\\mathbf{e}_k$\nmodality + time\nembeddings", fs=6.8)
arrow((xc, 36.5), side("tok", "l"))
box("enc", 41.0, 26.0, 16.5, 21.0,
    "Gap-aware selective\nstate-space encoder\n($L$ layers)\n\n$\\bar{\\mathbf{a}}_k=e^{-\\boldsymbol{\\lambda}(\\Delta t_k+\\delta_k)}$\n$\\mathbf{h}_k=\\bar{\\mathbf{a}}_k\\odot\\mathbf{h}_{k-1}+\\bar{\\mathbf{b}}_k\\odot\\mathbf{u}_k$",
    fs=6.8)
arrow(side("tok", "r"), side("enc", "l", 0.4762))
box("lm", 61.0, 33.0, 10.5, 7.0, "Landmark\nstate $\\mathbf{r}(\\tau)$", fs=6.8)
arrow(side("enc", "r", 0.4762), side("lm", "l"))

# ---------------- heads
box("anc", 61.0, 45.0, 17.5, 8.5, "Physiology anchor $\\mathbf{z}$\nCWSI, $\\chi$ (red-edge), $\\pi$", fill="0.93", fs=6.6)
box("cr", 75.5, 31.0, 23.0, 11.0, "Competing-risk hazard head\n$p_h(j\\mid\\mathbf{r},\\mathbf{z})$,  $j\\in\\{\\mathrm{W},\\mathrm{N},\\mathrm{P}\\}$\n$F_j(h)$, $h=1,\\ldots,H$", fs=6.6)
arrow(side("lm", "t"), (side("lm", "t")[0], 45.0))
arrow(side("lm", "r"), side("cr", "l", 0.3636))
# anchor -> hazard (signed link)
poly([side("anc", "r"), (87.0, side("anc", "r")[1]), (87.0, 42.0)])
ax.text(88.0, 47.0, "signed\nlink", fontsize=6.2, ha="left", va="center")

# ---------------- decision layer
box("lpb", 75.5, 17.5, 23.0, 8.0, "Shift-weighted conformal\nlead-time bound $L_\\alpha(\\mathbf{x})$", fs=6.6)
box("crc", 75.5, 5.5, 23.0, 8.0, "Risk-controlled alert\n$R_\\tau\\geq\\hat{\\theta}$, missed onsets $\\leq\\alpha_m$", fs=6.6)
arrow(side("cr", "b"), side("lpb", "t"))
arrow(side("lpb", "b"), side("crc", "t"))

# ---------------- training objectives (bottom) with dashed gradient paths
box("lmf", 23.5, 13.5, 14.2, 8.0, "Masked cross-modal\nforecasting $\\mathcal{L}_{\\mathrm{MF}}$", fill="0.86", fs=6.6)
box("lanc", 41.0, 13.5, 14.2, 8.0, "Predictive anchoring\n$\\mathcal{L}_{\\mathrm{anc}}$", fill="0.86", fs=6.6)
box("lcr", 58.5, 13.5, 12.6, 8.0, "Competing-risk\nlikelihood $\\mathcal{L}_{\\mathrm{CR}}$", fill="0.86", fs=6.6)
box("fed", 23.5, 2.0, 22.0, 7.5, "Federated averaging of all\nparameters $\\Theta$ (data stay on farm)", fs=6.6, ls=(0, (1.5, 1.2)))
box("kd", 49.0, 2.0, 21.0, 7.5, "Distilled edge student\n$\\mathcal{L}_{\\mathrm{KD}}=\\mathrm{KL}(p^{T}\\Vert p^{S})$", fs=6.6, ls=(0, (1.5, 1.2)))
# dashed reverse-mode gradient arrows into the encoder (and the anchor)
arrow(side("lmf", "t"), (side("lmf", "t")[0], 30.0), dashed=True)           # into tokenizer/encoder stack
arrow(side("lanc", "t"), (side("lanc", "t")[0], 26.0), dashed=True)
poly([side("lcr", "t"), (side("lcr", "t")[0], 24.0), (55.0, 24.0), (55.0, 26.0)], dashed=True)
# hazard output feeds the likelihood and, as teacher probabilities, the distilled student (forward, solid)
xv = 74.1
yh = side("cr", "l", 0.1)[1]
ax.plot([side("cr", "l", 0.1)[0], xv], [yh, yh], color="black", lw=0.8)
ax.plot([xv, xv], [yh, side("kd", "r")[1]], color="black", lw=0.8)
arrow((xv, side("lcr", "r", 0.6875)[1]), side("lcr", "r", 0.6875))
arrow((xv, side("kd", "r")[1]), side("kd", "r"))
ax.plot([xv], [side("lcr", "r", 0.6875)[1]], marker="o", ms=1.8, color="black")

# legend (placed in free space below the input column, no overlap)
lx = 1.0
for yy, lab, kind in [(6.9, "forward flow", "-"), (4.6, "gradient path", "--"), (2.3, "optional deployment", "box")]:
    if kind == "box":
        ax.add_patch(FancyBboxPatch((lx, yy - 0.75), 4.5, 1.5, boxstyle="round,pad=0.02,rounding_size=0.4",
                                    fc="white", ec="black", lw=0.8, ls=(0, (1.5, 1.2))))
    else:
        ax.plot([lx, lx + 4.5], [yy, yy], color="black", lw=0.8, ls=(0, (3.0, 2.0)) if kind == "--" else "-")
    ax.text(lx + 5.2, yy, lab, fontsize=6.2, va="center")
fig.savefig("figures/fig1_architecture.png", dpi=300)
print("ok")
