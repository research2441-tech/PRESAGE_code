import numpy as np


def temporal_split(S):
    y = np.array([s["year"] for s in S])
    return dict(train=np.where(y <= 2010)[0], val=np.where((y >= 2011) & (y <= 2013))[0],
                cal=np.where((y >= 2014) & (y <= 2017))[0], test=np.where(y >= 2018)[0])


def lozo_split(S, zone, seed=0):
    """leave-one-zone-out: test = all seasons of the held-out zone; remaining zones split by station
    into train / val / calibration so that no station contributes to two partitions"""
    rng = np.random.default_rng(100 + zone)
    z = np.array([s["zone"] for s in S]); st = np.array([s["code"] for s in S])
    test = np.where(z == zone)[0]
    rest = np.where(z != zone)[0]
    stations = np.unique(st[rest]); rng.shuffle(stations)
    n = len(stations); nv = max(2, int(0.15 * n)); nc = max(3, int(0.2 * n))
    vs, cs = set(stations[:nv]), set(stations[nv:nv + nc])
    val = np.array([i for i in rest if st[i] in vs]); cal = np.array([i for i in rest if st[i] in cs])
    train = np.array([i for i in rest if st[i] not in vs and st[i] not in cs])
    return dict(train=train, val=val, cal=cal, test=test)
