"""Training / inference utilities for PRESAGE and the sequence baselines."""
import copy
import time
import numpy as np
import torch
from presage_lib import (PRESAGE, Prep, collate, cr_nll, cif_from_logits, distill_loss, H, K)

torch.set_num_threads(2)


def batches(idx, enc, seasons, bs, rng, shuffle=True, **kw):
    idx = np.array(idx)
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    # length bucketing for speed
    chunks = [idx[i:i + bs * 8] for i in range(0, len(idx), bs * 8)]
    out = []
    for c in chunks:
        c = sorted(c, key=lambda i: len(enc[i]["t"]))
        out += [c[i:i + bs] for i in range(0, len(c), bs)]
    if shuffle:
        rng.shuffle(out)
    for b in out:
        yield b, collate([enc[i] for i in b], [seasons[i] for i in b], rng=rng, **kw)


def pretrain(model, tr, enc, seasons, epochs=6, bs=32, lr=2e-3, seed=0, log=print):
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(epochs):
        tot, nb, t0 = 0.0, 0, time.time()
        model.train()
        for _, b in batches(tr, enc, seasons, bs, rng, window_drop=0.8, token_drop=0.1):
            y = model.enc(b)
            loss = model.forecast_loss(b, y) + (0.5 * model.anchor_loss(b, y) if model.anchor else 0.0)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        log(f"  pre ep{ep} loss {tot / nb:.4f} {time.time() - t0:.0f}s")
    return model


def evaluate_nll(model, idx, enc, seasons, bs=64):
    model.eval(); tot, n = 0.0, 0
    rng = np.random.default_rng(123)
    with torch.no_grad():
        for _, b in batches(idx, enc, seasons, bs, rng, shuffle=False):
            _, lg, _ = model(b)
            if len(lg):
                tot += cr_nll(lg, b["delta"], b["cause"]).item() * len(lg); n += len(lg)
    return tot / max(n, 1)


def finetune(model, tr, va, enc, seasons, epochs=14, bs=32, lr=1.5e-3, seed=0, aux=True, patience=4,
             teacher=None, log=print):
    rng = np.random.default_rng(seed + 7)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train(); t0 = time.time(); tot, nb = 0.0, 0
        for bidx, b in batches(tr, enc, seasons, bs, rng, window_drop=0.3, token_drop=0.05):
            y, lg, z = model(b)
            loss = cr_nll(lg, b["delta"], b["cause"])
            if aux:
                loss = loss + 0.2 * model.forecast_loss(b, y)
                if model.anchor:
                    loss = loss + 0.2 * model.anchor_loss(b, y)
            if teacher is not None:
                with torch.no_grad():
                    _, tlg, _ = teacher(b)
                loss = loss + 1.0 * distill_loss(lg, tlg)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        v = evaluate_nll(model, va, enc, seasons)
        log(f"  ft ep{ep} train {tot / nb:.4f} val {v:.4f} {time.time() - t0:.0f}s")
        if v < best - 1e-4:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def predict(model, idx, enc, seasons, bs=64, **kw):
    """returns per-landmark records: season index, landmark day, delta, cause, CIF (H,K), z"""
    model.eval(); rng = np.random.default_rng(99)
    S, LM, D, C, CIF, Z = [], [], [], [], [], []
    with torch.no_grad():
        for bidx, b in batches(idx, enc, seasons, bs, rng, shuffle=False, **kw):
            _, lg, z = model(b)
            cif = cif_from_logits(lg).numpy()
            sidx = np.array(bidx)[b["lb_b"].numpy()]
            S.append(sidx); LM.append(b["lm_day"].numpy()); D.append(b["delta"].numpy()); C.append(b["cause"].numpy())
            CIF.append(cif); Z.append(z.numpy())
    return dict(season=np.concatenate(S), lm=np.concatenate(LM), delta=np.concatenate(D),
                cause=np.concatenate(C), cif=np.concatenate(CIF), z=np.concatenate(Z))
