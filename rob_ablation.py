"""Robustness of ablated variants (inference only, saved weights): does elapsed-time discretization or anchoring
help when modalities disappear?"""
import json, pickle, sys
import numpy as np, torch
import presage_lib as PL
from run_exp import build_model, ROB
from train import predict
from splits import temporal_split
import eval_lib as E
torch.set_num_threads(1)
S = pickle.load(open("data/sim_fieldseasons.pkl", "rb"))
sp = temporal_split(S)
prep = PL.Prep().fit([S[i] for i in sp["train"]]); enc = [prep.encode(s) for s in S]
conds = ["weather_only", "drop_soil", "drop_thermal", "loss50", "loss70"]
out = {}
for v in ["presage", "nogap", "noanchor", "nopre", "nosel"]:
    for seed in [0, 1]:
        m = build_model(v, 64); m.load_state_dict(torch.load(f"models/sim_temporal_{v}_{seed}.pt")); m.eval()
        for c in conds:
            p = predict(m, sp["test"], enc, S, **ROB[c])
            out.setdefault(v, {}).setdefault(c, []).append(E.point_metrics(p)["cindex"])
    print(v, {c: round(float(np.mean(x)), 4) for c, x in out[v].items()}, flush=True)
json.dump({v: {c: [float(np.mean(x)), float(np.std(x))] for c, x in d.items()} for v, d in out.items()},
          open("results/rob_ablation.json", "w"), indent=1)
