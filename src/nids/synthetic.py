"""Synthetic NF-v3-schema flows for smoke runs and tests. Never a result.

Each attack class has a distinct temporal signature, so the pipeline and the
spectral predictor can be exercised end to end:
  dos      large persistent offset in bytes/packets (strong DC energy)
  beacon   regular periodic flows with *lower* variance than benign
  scan     fast alternation (energy at high frequencies)
  slow     small persistent offset (absorbable by an adaptive equilibrium)
Attack episodes recur through the whole time span, so every class appears in
every temporal split. TTL columns leak the label on purpose, one column is
constant, one duplicates another, and a few values are inf/NaN, to exercise
the cleaner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import schema

ATTACKS = ("dos", "beacon", "scan", "slow")


def generate(n_flows: int = 60_000, hours: int = 12, seed: int = 0,
             attacks=ATTACKS, attack_frac: float = 0.3, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000_000.0
    t = np.sort(rng.uniform(0, hours * 3.6e6, n_flows)) + t0
    label = np.array([schema.BENIGN] * n_flows, dtype=object)
    # attack episodes: contiguous runs of 64..512 flows, classes cycled so that
    # every class recurs across the whole span (and so in every temporal split)
    n_ep = max(len(attacks) * 10, int(attack_frac * n_flows / 288))
    slots = np.linspace(0, n_flows, n_ep + 1).astype(int)
    for e in range(n_ep):
        L = int(rng.integers(64, 512))
        a0 = int(rng.integers(slots[e], max(slots[e] + 1, slots[e + 1] - L)))
        label[a0:a0 + L] = attacks[e % len(attacks)]
    idx = np.arange(n_flows)
    base = rng.lognormal(6 + shift, 1.0, n_flows)
    pk = rng.poisson(8, n_flows) + 1.0
    dur = rng.exponential(200, n_flows)
    for a in attacks:
        m = label == a
        k = idx[m]
        if a == "dos":
            base[m] *= 30
            pk[m] *= 20
        elif a == "beacon":
            base[m] = 400 + 5 * np.sin(2 * np.pi * k / 16) + rng.normal(0, 1, m.sum())
            pk[m] = 4
            dur[m] = 50
        elif a == "scan":
            base[m] = np.where(k % 2 == 0, 60.0, 4000.0) * rng.uniform(0.9, 1.1, m.sum())
            pk[m] = np.where(k % 2 == 0, 1.0, 12.0)
        elif a == "slow":
            base[m] *= 2.5
            dur[m] *= 3
    df = pd.DataFrame({c: np.zeros(n_flows, np.float32) for c in schema.EXPECTED_FEATURES})
    df["IPV4_SRC_ADDR"] = pd.Categorical(["10.0.0.%d" % (i % 50) for i in idx])
    df["IPV4_DST_ADDR"] = pd.Categorical(["10.0.1.%d" % (i % 20) for i in idx])
    df["L4_SRC_PORT"] = rng.integers(1024, 65535, n_flows).astype(np.float32)
    df["L4_DST_PORT"] = rng.choice([80, 443, 53, 22], n_flows).astype(np.float32)
    df["PROTOCOL"] = rng.choice([6, 17], n_flows).astype(np.float32)
    df["L7_PROTO"] = rng.choice([7.0, 91.0, 5.0], n_flows).astype(np.float32)
    df["IN_BYTES"] = base.astype(np.float32)
    df["OUT_BYTES"] = (base * rng.uniform(0.5, 2, n_flows)).astype(np.float32)
    df["IN_PKTS"] = pk.astype(np.float32)
    df["OUT_PKTS"] = (pk * rng.uniform(0.5, 1.5, n_flows)).astype(np.float32)
    df["FLOW_DURATION_MILLISECONDS"] = dur.astype(np.float32)
    df["DURATION_IN"] = dur.astype(np.float32)                 # identical to duration -> corr drop
    df["DURATION_OUT"] = (dur * rng.uniform(0.2, 0.6, n_flows)).astype(np.float32)
    df["TCP_FLAGS"] = rng.choice([2, 18, 24, 27], n_flows).astype(np.float32)
    df["MIN_TTL"] = np.where(label == schema.BENIGN, 64, 63).astype(np.float32)  # leaky
    df["MAX_TTL"] = df["MIN_TTL"]
    df["SRC_TO_DST_IAT_AVG"] = rng.exponential(10, n_flows).astype(np.float32)
    df["SRC_TO_DST_SECOND_BYTES"] = (base / (dur / 1000 + 1)).astype(np.float32)
    df.loc[rng.integers(0, n_flows, 5), "SRC_TO_DST_SECOND_BYTES"] = np.inf
    df.loc[rng.integers(0, n_flows, 5), "DNS_TTL_ANSWER"] = np.nan
    df["DNS_QUERY_ID"] = rng.integers(0, 65535, n_flows).astype(np.float32)
    df[schema.TIME_START] = t
    df[schema.TIME_END] = t + dur
    df[schema.LABEL_BIN] = (label != schema.BENIGN).astype(np.int8)
    df[schema.LABEL_MULTI] = pd.Categorical(label.astype(str))
    return df
