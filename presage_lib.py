"""PRESAGE: tokenisation, gap-aware selective state-space encoder, physiology-anchored competing-risk head,
self-supervised cross-modal forecasting, and baseline sequence encoders (GRU-D, time-aware Transformer)."""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = 8
K = 3          # causes: 0 water, 1 nitrogen, 2 pathogen
H = 21         # forecast horizon (days)
MODS = 7       # static, weather, soil, optical, thermal, mgmt, spore trap
DIMS = {0: 6, 1: 8, 2: 1, 3: 2, 4: 3, 5: 2, 6: 1}
LINK = [(0, 1.0), (1, -1.0), (2, 1.0)]  # cause j <- (anchor channel, physiological sign)
PROXY_MODE = "sim"


# ----------------------------------------------------------------------------- data
def season_arrays(s):
    T = len(s["tok_t"])
    v = np.zeros((T, P), np.float32)
    for i, x in enumerate(s["tok_v"]):
        v[i, :len(x)] = x
    return s["tok_t"].astype(np.float32), s["tok_m"].astype(np.int64), v


def temp_resp(t, tmin=10., topt=25., tmax=35.):
    t = np.clip(t, tmin, tmax)
    a = (t - tmin) / (topt - tmin); b = (tmax - t) / (tmax - topt)
    return np.clip(a * b ** ((tmax - topt) / (topt - tmin)), 0, 1)


class Prep:
    """Normalisation statistics, CWSI baseline and physics proxies, all fitted on training seasons only."""

    def fit(self, seasons):
        allv = {m: [] for m in range(MODS)}
        th = []
        for s in seasons:
            t, m, v = season_arrays(s)
            for k in range(MODS):
                if (m == k).any():
                    allv[k].append(v[m == k])
            if (m == 4).any():
                th.append(v[m == 4])
        self.mu = np.zeros((MODS, P), np.float32); self.sd = np.ones((MODS, P), np.float32)
        for k in range(MODS):
            if allv[k]:
                a = np.concatenate(allv[k])
                self.mu[k] = a.mean(0); self.sd[k] = a.std(0) + 1e-3
        self.mu[0] = 0; self.sd[0] = 1  # static kept raw
        # non-water-stressed baseline dT = a + b VPD from the lower 10% envelope (practitioner procedure)
        if not th:
            self.a_ll, self.b_ll, self.ul = 0.0, 0.0, 1.0
            return self
        th = np.concatenate(th)
        dT, vpd = th[:, 0], th[:, 2]
        bins = np.quantile(vpd, np.linspace(0, 1, 11))
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (vpd >= lo) & (vpd <= hi)
            if sel.sum() > 20:
                xs.append(vpd[sel].mean()); ys.append(np.quantile(dT[sel], 0.10))
        self.b_ll, self.a_ll = np.polyfit(xs, ys, 1)
        self.ul = float(np.quantile(dT, 0.90))
        return self

    def proxies(self, t, m, v):
        """physics proxies per token: [CWSI, chlorophyll ratio, 7-day infection pressure]; nan when absent"""
        prox = np.full((len(t), 3), np.nan, np.float32)
        if PROXY_MODE == "real":
            # [7-day atmospheric water deficit ET0 - P (mm/d), canopy moisture index NDMI, unused]
            we = np.where(m == 1)[0]
            if len(we):
                d = v[we, 6] - np.expm1(v[we, 3]); day = np.floor(t[we]); cum = np.cumsum(d)
                for i, dd in enumerate(day):
                    j = np.searchsorted(day, dd - 6, side="left")
                    prox[we[i], 0] = (cum[i] - (cum[j - 1] if j > 0 else 0)) / 7.0
            op = m == 3
            prox[op, 1] = v[op, 1]
            return prox
        th = m == 4
        ll = self.a_ll + self.b_ll * v[th, 2]
        prox[th, 0] = np.clip((v[th, 0] - ll) / np.maximum(self.ul - ll, 0.5), -0.3, 1.5)
        op = m == 3
        prox[op, 1] = np.clip(v[op, 1] / np.maximum(v[op, 0], 0.1), 0, 1.2)
        we = np.where(m == 1)[0]
        if len(we):
            tav = 0.5 * (v[we, 0] + v[we, 1]); rh = v[we, 2]; rain = np.expm1(v[we, 3])
            wet = np.where(rain >= 1.0, 1.0, np.clip((rh - 75.) / 15., 0, 1) * 0.6)
            p = temp_resp(tav) * wet
            day = np.floor(t[we])
            cum = np.cumsum(p)
            for i, d in enumerate(day):
                j = np.searchsorted(day, d - 6, side="left")
                prox[we[i], 2] = (cum[i] - (cum[j - 1] if j > 0 else 0)) / 7.0
        return prox

    def encode(self, s):
        t, m, v = season_arrays(s)
        prox = self.proxies(t, m, v)
        vn = (v - self.mu[m]) / self.sd[m]
        mask = np.zeros((len(t), P), np.float32)
        for k, d in DIMS.items():
            mask[m == k, :d] = 1
        vn = vn * mask
        return dict(t=t, m=m, v=vn, vmask=mask, prox=prox)


def landmark_targets(s):
    lms = np.array(s["landmarks"], np.int64)
    if s["T"] is None and s.get("cens") is not None:   # right-censored (end of record)
        delta = s["cens"] - lms; cause = np.full(len(lms), -2, np.int64)
    elif s["T"] is None:                                # season completed without onset
        delta = np.full(len(lms), 10 ** 4, np.int64); cause = np.full(len(lms), -1, np.int64)
    else:
        delta = s["T"] - lms; cause = np.full(len(lms), s["J"], np.int64)
    return lms, delta, cause


# ----------------------------------------------------------------------------- batching
def collate(encs, seasons, lm_sel=None, drop_mod=None, token_drop=0.0, window_drop=0.0, rng=None,
            n_fc=48, max_lb=7):
    """Pads token streams; optionally removes tokens (augmentation / robustness tests) and builds
    cross-modal forecasting targets from removed and retained tokens."""
    rng = rng or np.random.default_rng(0)
    items = []
    for e, s in zip(encs, seasons):
        t, m, v, vm, pr = e["t"], e["m"], e["v"], e["vmask"], e["prox"]
        keep = np.ones(len(t), bool)
        if drop_mod is not None:
            for dm in np.atleast_1d(drop_mod):
                keep &= m != dm
        if token_drop > 0:
            keep &= ~((rng.random(len(t)) < token_drop) & (m != 0))
        if window_drop > 0 and rng.random() < window_drop:
            mod = int(rng.choice([2, 3, 4, 6]))
            a = rng.uniform(0, max(1, t.max() - 30)); w = rng.uniform(15, 40)
            keep &= ~((m == mod) & (t >= a) & (t <= a + w))
        kt, km, kv, kvm, kpr = t[keep], m[keep], v[keep], vm[keep], pr[keep]
        # forecasting targets: any non-static/non-mgmt token (removed tokens favoured)
        cand = np.where(((m >= 1) & (m <= 4)) | (m == 6))[0]
        rem = cand[~keep[cand]]
        nrem = min(len(rem), n_fc // 2)
        pick = list(rng.choice(rem, nrem, replace=False)) if nrem else []
        pick += list(rng.choice(cand, min(len(cand), n_fc - nrem), replace=False))
        pick = np.array(pick, np.int64)
        lb = rng.uniform(0, max_lb, len(pick)).astype(np.float32)
        src = np.searchsorted(kt, t[pick] - lb, side="left") - 1  # last retained token strictly before t - lb
        ok = src >= 0
        pick, src = pick[ok], src[ok]
        lms, delta, cause = landmark_targets(s)
        if lm_sel is not None:
            sel = lm_sel(lms)
            lms, delta, cause = lms[sel], delta[sel], cause[sel]
        lidx = np.searchsorted(kt, lms + 1.0, side="left") - 1
        lastm = np.full((len(lms), MODS), -1, np.int64)
        for mm_ in range(1, MODS):
            pos = np.where(km == mm_)[0]
            if len(pos):
                j = np.searchsorted(kt[pos], lms + 1.0, side="left") - 1
                lastm[:, mm_] = np.where(j >= 0, pos[np.maximum(j, 0)], -1)
        items.append(dict(lastm=lastm, t=kt, m=km, v=kv, prox=kpr, fc_src=src, fc_m=m[pick], fc_v=v[pick], fc_vm=vm[pick],
                          fc_dt=t[pick] - kt[src], lidx=np.maximum(lidx, 0), lms=lms, delta=delta, cause=cause))
    B = len(items); T = max(len(x["t"]) for x in items)
    tt = np.zeros((B, T), np.float32); mm = np.zeros((B, T), np.int64); vv = np.zeros((B, T, P), np.float32)
    pad = np.zeros((B, T), bool); prox = np.full((B, T, 3), np.nan, np.float32)
    for i, x in enumerate(items):
        n = len(x["t"]); tt[i, :n] = x["t"]; mm[i, :n] = x["m"]; vv[i, :n] = x["v"]; pad[i, n:] = True
        prox[i, :n] = x["prox"]
        if n < T:
            tt[i, n:] = x["t"][-1]
    dt = np.diff(tt, axis=1, prepend=tt[:, :1]); dt = np.clip(dt, 0, None)
    out = dict(t=torch.tensor(tt), m=torch.tensor(mm), v=torch.tensor(vv), pad=torch.tensor(pad),
               dt=torch.tensor(dt), prox=torch.tensor(prox))
    # flattened landmark and forecasting indices
    lb_b, lb_i, dl, cs, lm_day, lastm = [], [], [], [], [], []
    fb, fi, fm, fv, fvm, fdt = [], [], [], [], [], []
    for i, x in enumerate(items):
        lb_b += [i] * len(x["lidx"]); lb_i += list(x["lidx"]); dl += list(x["delta"]); cs += list(x["cause"])
        lm_day += list(x["lms"]); lastm.append(x["lastm"])
        fb += [i] * len(x["fc_src"]); fi += list(x["fc_src"]); fm += list(x["fc_m"]); fv.append(x["fc_v"])
        fvm.append(x["fc_vm"]); fdt += list(x["fc_dt"])
    out.update(lb_b=torch.tensor(lb_b, dtype=torch.long), lb_i=torch.tensor(lb_i, dtype=torch.long),
               delta=torch.tensor(dl, dtype=torch.long), cause=torch.tensor(cs, dtype=torch.long),
               lm_day=torch.tensor(lm_day, dtype=torch.float32),
               lastm=torch.tensor(np.concatenate(lastm) if lastm else np.zeros((0, MODS)), dtype=torch.long),
               fb=torch.tensor(fb, dtype=torch.long), fi=torch.tensor(fi, dtype=torch.long),
               fm=torch.tensor(fm, dtype=torch.long),
               fv=torch.tensor(np.concatenate(fv) if fv else np.zeros((0, P)), dtype=torch.float32),
               fvm=torch.tensor(np.concatenate(fvm) if fvm else np.zeros((0, P)), dtype=torch.float32),
               fdt=torch.tensor(fdt, dtype=torch.float32))
    return out


# ----------------------------------------------------------------------------- encoders
def fourier(x, n=8, maxp=200.0):
    freqs = torch.exp(torch.linspace(math.log(1.0), math.log(maxp), n, device=x.device))
    a = x[..., None] * (2 * math.pi / freqs)
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class TokenEmbed(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(P, D) for _ in range(MODS)])
        self.memb = nn.Embedding(MODS, D)
        self.temb = nn.Linear(32, D)
        self.ln = nn.LayerNorm(D)

    def forward(self, b):
        v, m, t, dt = b["v"], b["m"], b["t"], b["dt"]
        x = torch.zeros(*v.shape[:2], self.memb.embedding_dim)
        for k in range(MODS):
            sel = m == k
            if sel.any():
                x[sel] = self.proj[k](v[sel])
        tf = torch.cat([fourier(t), fourier(torch.log1p(dt), maxp=8.0)], -1)
        return self.ln(x + self.memb(m) + self.temb(tf))


def seq_scan(log_a, u):
    """sequential form of the same recurrence (faster on CPU for short sequences)"""
    a = torch.exp(log_a).unbind(1)
    uu = u.unbind(1)
    B, T, D, N = u.shape
    h = torch.zeros(B, D, N); hs = []
    for t in range(T):
        h = a[t] * h + uu[t]
        hs.append(h)
    return torch.stack(hs, 1)


def chunk_scan(log_a, u, Q=16):
    """h_t = exp(log_a_t) * h_{t-1} + u_t over dim 1, computed chunk-wise with stable segment sums.
    log_a, u: (B, T, D, N)"""
    B, T, D, N = u.shape
    pad = (-T) % Q
    if pad:
        log_a = F.pad(log_a, (0, 0, 0, 0, 0, pad)); u = F.pad(u, (0, 0, 0, 0, 0, pad))
    C = log_a.shape[1] // Q
    la = log_a.view(B, C, Q, D, N).permute(0, 1, 3, 4, 2)  # B C D N Q
    uu = u.view(B, C, Q, D, N).permute(0, 1, 3, 4, 2)
    tri = torch.tril(torch.ones(Q, Q, dtype=torch.bool))
    h = torch.zeros(B, D, N)
    outs = []
    for c in range(C):
        cum = torch.cumsum(la[:, c], -1)  # B D N Q
        seg = cum[..., :, None] - cum[..., None, :]
        seg = seg.masked_fill(~tri, float("-inf"))
        Lm = torch.exp(seg)
        hc = torch.einsum("bdnts,bdns->bdnt", Lm, uu[:, c]) + torch.exp(cum) * h[..., None]
        outs.append(hc)
        h = hc[..., -1]
    hs = torch.stack(outs, 1).permute(0, 1, 4, 2, 3).reshape(B, C * Q, D, N)
    return hs[:, :T]


class GapSSM(nn.Module):
    """Selective diagonal state-space layer whose discretisation uses the real elapsed time between tokens.
    gap_aware=False replaces elapsed time by a unit step (ordinal sequence); selective=False fixes Delta."""

    def __init__(self, D, N=8, gap_aware=True, selective=True, expand=2):
        super().__init__()
        E = expand * D
        self.inp = nn.Linear(D, 2 * E)
        self.dlt = nn.Linear(E, E)
        self.Bp = nn.Linear(E, N); self.Cp = nn.Linear(E, N)
        tau = torch.exp(torch.linspace(math.log(0.7), math.log(60.0), N))  # memory time constants (days)
        self.log_lam = nn.Parameter(torch.log(1.0 / tau).repeat(E, 1))
        self.Dskip = nn.Parameter(torch.ones(E))
        self.out = nn.Linear(E, D)
        self.ln = nn.LayerNorm(D)
        self.gap_aware, self.selective = gap_aware, selective
        self.fixed_delta = nn.Parameter(torch.zeros(E))

    def forward(self, x, dt, pad):
        r = x
        xz = self.inp(self.ln(x)); xi, z = xz.chunk(2, -1)
        xi = F.silu(xi)
        delta = F.softplus(self.dlt(xi) if self.selective else self.fixed_delta.expand_as(xi)) + 1e-3
        lam = torch.exp(self.log_lam)  # E N
        gap = dt if self.gap_aware else (dt > 0).float() + 0.0 * dt
        log_a = -lam * (gap[..., None, None] + delta[..., None])
        beta = (1 - torch.exp(-lam * delta[..., None])) / lam  # zero-order-hold input gain
        u = beta * xi[..., None] * self.Bp(xi)[:, :, None, :]
        pm = pad[..., None, None]
        log_a = log_a.masked_fill(pm, 0.0); u = u.masked_fill(pm, 0.0)
        h = seq_scan(log_a, u)
        y = (h * self.Cp(xi)[:, :, None, :]).sum(-1) + self.Dskip * xi
        y = y * F.silu(z)
        return r + self.out(y)


class GapSSMEncoder(nn.Module):
    def __init__(self, D=48, N=8, L=2, gap_aware=True, selective=True):
        super().__init__()
        self.emb = TokenEmbed(D)
        self.layers = nn.ModuleList([GapSSM(D, N, gap_aware, selective) for _ in range(L)])
        self.lnf = nn.LayerNorm(D)

    def forward(self, b):
        x = self.emb(b)
        for l in self.layers:
            x = l(x, b["dt"], b["pad"])
        return self.lnf(x)


class TimeTransformerEncoder(nn.Module):
    """Causal Transformer over the same tokens with continuous-time embeddings (attention baseline)."""

    def __init__(self, D=48, L=2, heads=4):
        super().__init__()
        self.emb = TokenEmbed(D)
        layer = nn.TransformerEncoderLayer(D, heads, 2 * D, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L)
        self.lnf = nn.LayerNorm(D)

    def forward(self, b):
        x = self.emb(b)
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        # tokens may attend to tokens with equal or earlier time stamps only (no future leakage)
        y = self.enc(x, mask=causal, src_key_padding_mask=b["pad"])
        return self.lnf(torch.nan_to_num(y))


class GRUDEncoder(nn.Module):
    """GRU-D style recurrent baseline: per-token update with input decay toward the empirical mean and
    hidden-state decay as a function of elapsed time."""

    def __init__(self, D=48):
        super().__init__()
        self.emb = TokenEmbed(D)
        self.cell = nn.GRUCell(D, D)
        self.gamma_h = nn.Linear(1, D)
        self.lnf = nn.LayerNorm(D)

    def forward(self, b):
        x = self.emb(b)
        B, T, D = x.shape
        h = torch.zeros(B, D); outs = []
        G = torch.exp(-F.relu(self.gamma_h(b["dt"][..., None]))).unbind(1)
        X = x.unbind(1); PADS = b["pad"][..., None].unbind(1)
        for i in range(T):
            hn = self.cell(X[i], h * G[i])
            keep = PADS[i]
            h = torch.where(keep, h, hn)
            outs.append(h)
        return self.lnf(torch.stack(outs, 1))


# ----------------------------------------------------------------------------- heads and model
class CompetingRiskHead(nn.Module):
    def __init__(self, D, anchor=True, ctx=0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(D + ctx, 2 * D), nn.GELU(), nn.Linear(2 * D, H * (K + 1)))
        self.anchor = anchor
        self.K = K
        self.gamma = nn.Parameter(torch.zeros(K))
        self.chan = [c for c, _ in LINK[:K]]
        self.sign = torch.tensor([s for _, s in LINK[:K]])

    def forward(self, h, z=None):
        lg = self.mlp(h).view(-1, H, self.K + 1)
        if self.anchor and z is not None:
            zz = torch.stack([z[:, c] if c is not None else torch.zeros_like(z[:, 0]) for c in self.chan], 1)
            link = self.sign * F.softplus(self.gamma) * zz  # (n, K)
            lg = lg + F.pad(link, (1, 0))[:, None, :]
        return lg


class PRESAGE(nn.Module):
    def __init__(self, D=48, N=8, L=2, encoder="ssm", anchor=True, gap_aware=True, selective=True, lastctx=False):
        super().__init__()
        self.lastctx = lastctx
        if encoder == "ssm":
            self.enc = GapSSMEncoder(D, N, L, gap_aware, selective)
        elif encoder == "transformer":
            self.enc = TimeTransformerEncoder(D, L)
        elif encoder == "grud":
            self.enc = GRUDEncoder(D)
        self.anchor = anchor
        self.zhead = nn.Linear(D, 3)
        self.cr = CompetingRiskHead(D, anchor, ctx=6 + 16 + ((MODS - 1) * (P + 2) if lastctx else 0))
        self.fc = nn.Sequential(nn.Linear(D + D + 16, 2 * D), nn.GELU(), nn.Linear(2 * D, P))
        self.fmemb = nn.Embedding(MODS, D)

    def forward(self, b):
        y = self.enc(b)
        hl = y[b["lb_b"], b["lb_i"]]
        z = self.zhead(hl)
        ctx = torch.cat([b["v"][b["lb_b"], 0, :6], fourier(b["lm_day"], 8, 200.0)], -1)  # static token + landmark time
        if self.lastctx:  # last observed value, presence and log-age of every modality at the landmark
            li = b["lastm"][:, 1:]; ok = (li >= 0).float()
            bb = b["lb_b"][:, None].expand_as(li)
            vals = b["v"][bb, li.clamp(min=0)] * ok[..., None]
            age = torch.log1p((b["lm_day"][:, None] + 1.0 - b["t"][bb, li.clamp(min=0)]).clamp(min=0)) * ok
            ctx = torch.cat([ctx, vals.flatten(1), ok, age], -1)
        logits = self.cr(torch.cat([hl, ctx], -1), z if self.anchor else None)
        return y, logits, z

    def forecast_loss(self, b, y):
        if len(b["fb"]) == 0:
            return torch.tensor(0.0)
        hs = y[b["fb"], b["fi"]]
        q = torch.cat([hs, self.fmemb(b["fm"]), fourier(b["fdt"], 8, 30.0)], -1)
        pred = self.fc(q)
        return (((pred - b["fv"]) ** 2) * b["fvm"]).sum() / b["fvm"].sum().clamp(min=1)

    def anchor_loss(self, b, y):
        """predictive anchoring: z read from the state just before a token must match that token's proxy"""
        z = self.zhead(y[:, :-1])
        tgt = b["prox"][:, 1:]
        ok = ~torch.isnan(tgt) & ~b["pad"][:, 1:, None]
        if ok.sum() == 0:
            return torch.tensor(0.0)
        return F.smooth_l1_loss(z[ok], torch.nan_to_num(tgt)[ok])


def cr_nll(logits, delta, cause):
    """discrete-time competing-risk negative log-likelihood; delta in days (>=1), cause -1 = no event"""
    logp = F.log_softmax(logits, -1)  # n H K+1
    n = logits.shape[0]
    hgrid = torch.arange(1, H + 1)[None, :]
    ev = (cause >= 0) & (delta <= H)
    surv_steps = torch.where(ev, delta - 1, torch.full_like(delta, H))
    surv_steps = torch.where(cause == -2, delta.clamp(max=H), surv_steps)  # censored before the horizon
    sm = (hgrid <= surv_steps[:, None]).float()
    ll = (logp[:, :, 0] * sm).sum(1)
    idx = (delta.clamp(1, H) - 1)
    ll_ev = logp[torch.arange(n), idx, (cause + 1).clamp(min=0)]
    ll = ll + torch.where(ev, ll_ev, torch.zeros_like(ll_ev))
    return -ll.mean()


def cif_from_logits(logits):
    p = torch.softmax(logits, -1)
    surv = torch.cumprod(p[:, :, 0], 1)
    prev = torch.cat([torch.ones_like(surv[:, :1]), surv[:, :-1]], 1)
    inc = prev[:, :, None] * p[:, :, 1:]
    return torch.cumsum(inc, 1)  # n H K


def distill_loss(student_logits, teacher_logits, T=1.0):
    return F.kl_div(F.log_softmax(student_logits / T, -1), F.softmax(teacher_logits / T, -1),
                    reduction="batchmean") / H
