"""One experimental run: tier (sim/real) x split x model variant x seed. Saves calibration/test predictions."""
import argparse
import copy
import json
import os
import pickle
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import presage_lib as PL
from presage_lib import PRESAGE, Prep
from train import pretrain, finetune, predict, evaluate_nll, batches
from splits import temporal_split, lozo_split

ROB = {"drop_thermal": dict(drop_mod=4), "drop_optical": dict(drop_mod=3), "drop_soil": dict(drop_mod=2),
       "drop_trap": dict(drop_mod=6), "weather_only": dict(drop_mod=[2, 3, 4, 5, 6]),
       "loss30": dict(token_drop=0.3), "loss50": dict(token_drop=0.5), "loss70": dict(token_drop=0.7)}


def get_data(tier):
    if tier == "sim":
        S = pickle.load(open(os.environ.get("SIMDATA", "data/sim_fieldseasons.pkl"), "rb"))
        if os.environ.get("SMOKE"):
            S = [s for s in S if s["zone"] in (0, 1)][:300] + [s for s in S if s["zone"] in (2, 3)][:300]
    else:
        S = pickle.load(open("data/real_episodes.pkl", "rb"))
        PL.K = 2
        PL.LINK = [(0, 1.0), (None, 0.0)]
        PL.PROXY_MODE = "real"
    return S


def get_split(tier, split, S, seed):
    if tier == "sim":
        if split == "temporal":
            return temporal_split(S)
        return lozo_split(S, int(split[-1]), seed)
    # real: leave-one-consortium-out; remaining consortia split by sensor stream
    c = int(split[-1]); rng = np.random.default_rng(200 + c)
    cons = np.array([s["cons"] for s in S]); code = np.array([s["code"] for s in S])
    test = np.where(cons == c)[0]; rest = np.where(cons != c)[0]
    streams = np.unique(code[rest]); rng.shuffle(streams)
    nv, nc = max(2, int(0.15 * len(streams))), max(3, int(0.25 * len(streams)))
    vs, cs = set(streams[:nv]), set(streams[nv:nv + nc])
    return dict(train=np.array([i for i in rest if code[i] not in vs | cs]),
                val=np.array([i for i in rest if code[i] in vs]),
                cal=np.array([i for i in rest if code[i] in cs]), test=test)


def build_model(variant, D, L=2, lastctx=0):
    kw = dict(D=D, L=L, lastctx=bool(lastctx))
    if variant == "noanchor":
        kw["anchor"] = False
    if variant == "nogap":
        kw["gap_aware"] = False
    if variant == "nosel":
        kw["selective"] = False
    if variant in ("transformer", "grud"):
        kw.update(encoder=variant, anchor=False)
    if variant == "student":
        kw.update(D=24, L=1)
    return PRESAGE(**kw)


def fedavg(model, clients, va, enc, S, rounds_pre=5, rounds_ft=20, seed=0, log=print):
    """FedAvg: each client runs one local epoch per round on its own seasons; weights averaged by size"""
    sizes = np.array([len(c) for c in clients], float); wts = sizes / sizes.sum()
    best, best_state, bad = 1e9, None, 0
    for phase, R in (("pre", rounds_pre), ("ft", rounds_ft)):
        for r in range(R):
            states = []
            for ci, cidx in enumerate(clients):
                local = copy.deepcopy(model)
                if phase == "pre":
                    pretrain(local, cidx, enc, S, epochs=1, seed=seed * 100 + r * 10 + ci, log=lambda *a: None)
                else:
                    local, _ = finetune(local, cidx, cidx[:1] if False else va, enc, S, epochs=1, seed=seed * 100 + r * 10 + ci,
                                        log=lambda *a: None, patience=99)
                states.append(local.state_dict())
            avg = {k: sum(w * s[k].float() for w, s in zip(wts, states)) for k in states[0]}
            model.load_state_dict(avg)
            if phase == "ft":
                v = evaluate_nll(model, va, enc, S)
                log(f"  fed round {r} val {v:.4f}")
                if v < best - 1e-4:
                    best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
                else:
                    bad += 1
                    if bad >= 5:
                        break
    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="sim"); ap.add_argument("--split", default="temporal")
    ap.add_argument("--model", default="presage"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--D", type=int, default=64); ap.add_argument("--pre", type=int, default=5)
    ap.add_argument("--ft", type=int, default=30); ap.add_argument("--rob", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--tag", default="")
    ap.add_argument("--tpre", type=int, default=0)  # 1: self-supervised pretraining also on unlabeled target-domain streams
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--lastctx", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs("results", exist_ok=True); os.makedirs("models", exist_ok=True)
    name = f"{a.tier}_{a.split}_{a.model}_{a.seed}{a.tag}"
    out = f"results/{name}.pkl"
    if os.path.exists(out):
        print("exists", out); return
    S = get_data(a.tier)
    sp = get_split(a.tier, a.split, S, a.seed)
    prep = Prep().fit([S[i] for i in sp["train"]])
    enc = [prep.encode(s) for s in S]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    logf = open(f"results/{name}.log", "w")
    log = lambda m: (print(m), logf.write(m + "\n"), logf.flush())
    model = build_model(a.model, a.D, a.L, a.lastctx)
    pre_idx = np.concatenate([sp["train"], sp["val"], sp["cal"], sp["test"]]) if a.tpre else sp["train"]
    if a.model == "fed":
        key = "zone" if a.tier == "sim" else "cons"
        grp = np.array([S[i][key] for i in sp["train"]])
        clients = [sp["train"][grp == g] for g in np.unique(grp)]
        model = fedavg(model, clients, sp["val"], enc, S, seed=a.seed, log=log)
    elif a.model == "local":
        # one model per client (zone); test seasons are scored by the model of their own zone
        key = "zone" if a.tier == "sim" else "cons"
        preds_cal, preds_test = [], []
        res = {}
        for g in np.unique([S[i][key] for i in sp["train"]]):
            m = build_model("presage", a.D, a.L, a.lastctx); torch.manual_seed(a.seed)
            tr = np.array([i for i in sp["train"] if S[i][key] == g])
            va = np.array([i for i in sp["val"] if S[i][key] == g]) if any(S[i][key] == g for i in sp["val"]) else tr[:20]
            pretrain(m, tr, enc, S, epochs=a.pre, seed=a.seed, log=log)
            m, _ = finetune(m, tr, va, enc, S, epochs=a.ft, seed=a.seed, log=log)
            ca = np.array([i for i in sp["cal"] if S[i][key] == g]); te = np.array([i for i in sp["test"] if S[i][key] == g])
            if len(ca): preds_cal.append(predict(m, ca, enc, S))
            if len(te): preds_test.append(predict(m, te, enc, S))
        cat = lambda L: {k: np.concatenate([p[k] for p in L]) for k in L[0]}
        pickle.dump(dict(cal=cat(preds_cal), test=cat(preds_test), time=time.time() - t0, args=vars(a)), open(out, "wb"))
        return
    else:
        if a.model == "student":
            teacher = build_model("presage", a.D, a.L, a.lastctx)
            teacher.load_state_dict(torch.load(f"models/{a.tier}_{a.split}_presage_{a.seed}{a.tag}.pt"))
            teacher.eval()
            model, _ = finetune(model, sp["train"], sp["val"], enc, S, epochs=a.ft, seed=a.seed, teacher=teacher, log=log)
        else:
            if a.model not in ("nopre", "transformer", "grud"):
                pretrain(model, pre_idx, enc, S, epochs=a.pre, seed=a.seed, log=log)
            aux = a.model not in ("transformer", "grud")
            model, _ = finetune(model, sp["train"], sp["val"], enc, S, epochs=a.ft, seed=a.seed, aux=aux, log=log)
    train_time = time.time() - t0
    torch.save(model.state_dict(), f"models/{name}.pt")
    res = dict(cal=predict(model, sp["cal"], enc, S), test=predict(model, sp["test"], enc, S), time=train_time,
               params=sum(p.numel() for p in model.parameters()), args=vars(a))
    # inference latency per landmark (single-thread CPU)
    torch.set_num_threads(1)
    t1 = time.time(); _ = predict(model, sp["test"][:40], enc, S, bs=1)
    nl = sum(len(S[i]["landmarks"]) for i in sp["test"][:40])
    res["latency_ms_per_season"] = 1000 * (time.time() - t1) / 40
    res["n_landmarks_latency"] = nl
    if a.rob:
        for k, kw in ROB.items():
            if a.tier == "real" and k in ("drop_thermal", "drop_trap"):
                continue
            res["rob_" + k] = predict(model, sp["test"], enc, S, **kw)
    pickle.dump(res, open(out, "wb"))
    log(f"done {name} {train_time:.0f}s")


if __name__ == "__main__":
    main()
