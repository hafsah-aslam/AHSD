"""NetFlow v3 (UQ NIDS) schema: expected columns and preprocessing roles.

Column names follow Luay et al., arXiv:2503.04404 (2025): the 43 NetFlow v2
features plus 10 temporal v3 features, plus the `Label` and `Attack` columns.
The audit records the real columns; `check_columns` reports any mismatch with
this list instead of silently assuming it.
"""

from __future__ import annotations

V2_FEATURES = [
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL",
    "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS",
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
]
V3_TEMPORAL = [
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    "SRC_TO_DST_IAT_MIN", "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV", "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG", "DST_TO_SRC_IAT_STDDEV",
]
EXPECTED_FEATURES = V2_FEATURES + V3_TEMPORAL  # 53
LABEL_BIN = "Label"
LABEL_MULTI = "Attack"
EXPECTED_COLUMNS = EXPECTED_FEATURES + [LABEL_BIN, LABEL_MULTI]
BENIGN = "Benign"

TIME_START = "FLOW_START_MILLISECONDS"
TIME_END = "FLOW_END_MILLISECONDS"

# Drop rules: column -> reason. Logged verbatim in PREPROCESS_LOG.md.
DROP_IDENTIFIERS = {
    "IPV4_SRC_ADDR": "identifier (host address); topology-specific, leaks across networks",
    "IPV4_DST_ADDR": "identifier (host address); topology-specific, leaks across networks",
    "L4_SRC_PORT": "identifier (ephemeral source port)",
    "L4_DST_PORT": "identifier (destination port)",
    "DNS_QUERY_ID": "identifier (random DNS transaction ID)",
}
DROP_TIMESTAMPS = {
    TIME_START: "timestamp; kept aside for sorting/grouping/splitting only",
    TIME_END: "timestamp; kept aside, never a feature",
}
DROP_TTL = {
    "MIN_TTL": "label-leakage artifact (testbed-specific TTLs); kept only in the TTL ablation",
    "MAX_TTL": "label-leakage artifact (testbed-specific TTLs); kept only in the TTL ablation",
}
DROP_LABELS = {
    LABEL_BIN: "label (binary)",
    LABEL_MULTI: "label (multiclass)",
}

# Non-negative heavy-tailed counts that get log1p (only if train min >= 0).
_HEAVY_TOKENS = ("BYTES", "PKTS", "DURATION", "THROUGHPUT", "IAT",
                 "FLOW_PKT", "PKT_LEN", "TCP_WIN_MAX", "DNS_TTL_ANSWER")


# D5 / Lycos2017 (lower-case LycoSTand names; AMENDMENT_04): flag counts, ip_prot and
# down_up_ratio are never logged.
_HEAVY_TOKENS_LYCOS = ("len", "cnt", "tot", "bytes", "per_s", "iat", "duration", "active", "idle", "bulk",
                       "subflow", "win", "var", "std", "mean", "max", "min")


def is_heavy_tailed(col: str) -> bool:
    if col.islower():  # LycoSTand naming (D5)
        if col.startswith(("flag_", "fwd_flag_", "bwd_flag_")) or col in ("ip_prot", "down_up_ratio"):
            return False
        return any(tok in col for tok in _HEAVY_TOKENS_LYCOS)
    return any(tok in col for tok in _HEAVY_TOKENS)


# Bookkeeping columns added by the pipeline; never features.
INTERNAL = {
    "row_id": "pipeline bookkeeping (raw row number)",
    "group": "pipeline bookkeeping (1-hour group id)",
    "cls": "pipeline bookkeeping (class code)",
}


def drop_map(keep_ttl: bool = False) -> dict[str, str]:
    d = {**DROP_IDENTIFIERS, **DROP_TIMESTAMPS, **DROP_LABELS, **INTERNAL}
    if not keep_ttl:
        d.update(DROP_TTL)
    return d


def check_columns(columns) -> dict:
    cols = list(columns)
    return {
        "missing": [c for c in EXPECTED_COLUMNS if c not in cols],
        "unexpected": [c for c in cols if c not in EXPECTED_COLUMNS],
        "n_columns": len(cols),
    }
