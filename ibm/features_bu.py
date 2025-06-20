import os
import pickle
import sys
import uuid

import numpy as np
import pandas as pd
import igraph as ig

from common import (
    load_dump, reset_multi_proc_staging, wait_for_processes,
    dump_object_for_proc, construct_arguments, collect_multi_proc_output
)


currency_rates = {
    "jpy": np.float32(0.009487665410827868),
    "cny": np.float32(0.14930721887033868),
    "cad": np.float32(0.7579775434031815),
    "sar": np.float32(0.2665884611958837),
    "aud": np.float32(0.7078143121927827),
    "ils": np.float32(0.29612081311363503),
    "chf": np.float32(1.0928961554056371),
    "usd": np.float32(1.0),
    "eur": np.float32(1.171783425225877),
    "rub": np.float32(0.012852809604990688),
    "gbp": np.float32(1.2916554735187644),
    "btc": np.float32(11879.132698717296),
    "inr": np.float32(0.013615817231245796),
    "mxn": np.float32(0.047296753463246695),
    "brl": np.float32(0.1771008654705292),
}

types = {
    "key": str,
    "num_source_or_target": np.uint16,
    "num_source_and_target": np.uint16,
    "num_source_only": np.uint16,
    "num_target_only": np.uint16,
    "num_transactions": np.uint16,
    "num_currencies": np.uint16,
    "num_source_or_target_bank": np.uint16,
    "num_source_and_target_bank": np.uint16,
    "num_source_only_bank": np.uint16,
    "num_target_only_bank": np.uint16,
    "turnover": np.uint64,
    "ts_range": np.uint32,
    "ts_std": np.float64,
    "ts_weighted_mean": np.float64,
    "ts_weighted_median": np.float64,
    "ts_weighted_std": np.float64,
    "assortativity_degree": np.float64,
    "assortativity_degree_ud": np.float64,
    "max_degree": np.uint16,
    "max_degree_in": np.uint16,
    "max_degree_out": np.uint16,
    "diameter": np.uint8,
    "assortativity_degree_bank": np.float64,
    "assortativity_degree_bank_ud": np.float64,
    "max_degree_bank": np.uint16,
    "max_degree_in_bank": np.uint16,
    "max_degree_out_bank": np.uint16,
    "diameter_bank": np.uint8,
    "usd": np.float32,
    "btc": np.float32,
    "chf": np.float32,
    "gbp": np.float32,
    "inr": np.float32,
    "jpy": np.float32,
    "rub": np.float32,
    "aud": np.float32,
    "mxn": np.float32,
    "ils": np.float32,
    "cad": np.float32,
    "brl": np.float32,
    "sar": np.float32,
    "cny": np.float32,
    "eur": np.float32,
}


def weighted_quantiles(values, weights, quantiles=0.5, interpolate=True):
    i = values.argsort()
    sorted_weights = weights[i]
    sorted_values = values[i]
    sorted_weights_cumsum = sorted_weights.cumsum()

    if interpolate:
        xp = (sorted_weights_cumsum - sorted_weights/2 ) / sorted_weights_cumsum[-1]
        return np.interp(quantiles, xp, sorted_values)
    else:
        return sorted_values[np.searchsorted(sorted_weights_cumsum, quantiles * sorted_weights_cumsum[-1])]


def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)


def get_segments(source_column, target_column, data_in):
    sources = set(data_in[source_column].unique())
    targets = set(data_in[target_column].unique())
    source_or_target = sources.union(targets)
    source_and_target = sources.intersection(targets)
    source_only = sources.difference(targets)
    target_only = targets.difference(sources)
    return source_or_target, source_and_target, source_only, target_only


def generate_features(df, group_id, graph_features=False):
    source_or_target, source_and_target, source_only, target_only = get_segments(
        "source", "target", df
    )
    features_row = {
        "key": group_id,
        "num_source_or_target": len(source_or_target),
        "num_source_and_target": len(source_and_target),
        "num_source_only": len(source_only),
        "num_target_only": len(target_only),
        "num_transactions": df.shape[0],
        "num_currencies": df["source_currency"].nunique(),
    }
    source_or_target, source_and_target, source_only, target_only = get_segments(
        "source_bank", "target_bank", df
    )
    features_row["num_source_or_target_bank"] = len(source_or_target)
    features_row["num_source_and_target_bank"] = len(source_and_target)
    features_row["num_source_only_bank"] = len(source_only)
    features_row["num_target_only_bank"] = len(target_only)

    left = (
        df.loc[:, ["target", "source_currency", "source_amount"]]
        .rename(columns={"target": "source"})
        .groupby(["source", "source_currency"])
        .agg({"source_amount": "sum"})
    )
    right = df.groupby(["source", "source_currency"]).agg({"source_amount": "sum"})
    result = left.join(right, how="outer", lsuffix="_left").fillna(0).reset_index()
    result.loc[:, "delta"] = result["source_amount_left"] - result["source_amount"]
    turnover_currency = result[result["delta"] > 0].reset_index(drop=True)
    turnover_currency = (
        turnover_currency.groupby("source_currency").agg({"delta": "sum"}).to_dict()["delta"]
    )

    left = (
        df.loc[:, ["target", "amount"]]
        .rename(columns={"target": "source"})
        .groupby("source")
        .agg({"amount": "sum"})
    )
    right = df.groupby("source").agg({"amount": "sum"})
    result = left.join(right, how="outer", lsuffix="_left").fillna(0).reset_index()
    result.loc[:, "delta"] = result["amount_left"] - result["amount"]
    turnover = float(result[result["delta"] > 0]["delta"].sum())
    turnover_currency_norm = {}
    for key, value in turnover_currency.items():
        turnover_currency_norm[key] = float((currency_rates[key] * value) / turnover)

    features_row["turnover"] = turnover
    features_row.update(turnover_currency_norm)

    features_row["ts_range"] = df["window_delta"].max() - df["window_delta"].min()
    features_row["ts_std"] = df["window_delta"].std()
    features_row["ts_weighted_mean"] = np.average(df["window_delta"], weights=df["amount"])
    features_row["ts_weighted_median"] = weighted_quantiles(
        df["window_delta"].values, weights=df["amount"].values, quantiles=0.5, interpolate=True
    )
    features_row["ts_weighted_std"] = weighted_std(df["window_delta"], df["amount"])

    if graph_features:
        graph = ig.Graph.DataFrame(df[["source", "target"]], use_vids=False, directed=True)
        features_row["assortativity_degree"] = graph.assortativity_degree(directed=True)
        features_row["assortativity_degree_ud"] = graph.assortativity_degree(directed=False)
        features_row["max_degree"] = max(graph.degree(mode="all"))
        features_row["max_degree_in"] = max(graph.degree(mode="in"))
        features_row["max_degree_out"] = max(graph.degree(mode="out"))
        features_row["diameter"] = graph.diameter(directed=True, unconn=True)
    
        graph = ig.Graph.DataFrame(
            df[["source_bank", "target_bank"]], use_vids=False, directed=True
        )
        features_row["assortativity_degree_bank"] = graph.assortativity_degree(
            directed=True
        )
        features_row["assortativity_degree_bank_ud"] = graph.assortativity_degree(
            directed=False
        )
        features_row["max_degree_bank"] = max(graph.degree(mode="all"))
        features_row["max_degree_in_bank"] = max(graph.degree(mode="in"))
        features_row["max_degree_out_bank"] = max(graph.degree(mode="out"))
        features_row["diameter_bank"] = graph.diameter(directed=True, unconn=True)

    return features_row

def get_features_chunk(df_chunk, graph_features):
    features_all = []
    if isinstance(df_chunk, pd.DataFrame):
        df_chunk = df_chunk.groupby("id")
    for key_, group in df_chunk:
        features_all.append(generate_features(group, key_, graph_features=graph_features))
    features_all = pd.DataFrame(features_all)
    found_types = {k: v for k, v in types.items() if k in features_all.columns}
    dump_object_for_proc(features_all.astype(found_types), False, pandas=True)


def get_features_chunk_with_gf(chunk_loc):
    return get_features_chunk(pd.read_parquet(chunk_loc), True)


def get_features_chunk_without_gf(chunk_loc):
    return get_features_chunk(load_dump(chunk_loc), False)


def get_features_multi_proc(chunks_locations, proc, reset_staging=False):
    if reset_staging:
        reset_multi_proc_staging()
    process_ids = set()
    for chunk_location in chunks_locations:
        process_id = str(uuid.uuid4())
        args = construct_arguments(chunk_location)
        os.system(f"{sys.executable} spawn.py {process_id} {proc} {args} &")
        process_ids = process_ids.union([process_id])
    wait_for_processes(process_ids)
    return collect_multi_proc_output(pandas=True)


def pov_features(group_id, group):
    src, tgt = group_id
    currency_turnover = (
        group
        .groupby("source_currency")
        .agg({"source_amount": "sum"})
    ).to_dict()["source_amount"]
    total = group["amount"].sum()
    currency_turnover = {k: v / total for k, v in currency_turnover.items()}
    row = {"source": src, "target": tgt, "total_amount": total}
    row.update(currency_turnover)
    format_turnover = (
        group
        .groupby("format")
        .agg({"amount": "sum"})
    ).to_dict()["amount"]
    format_turnover = {k.lower().replace(" ", "_"): v / total for k, v in format_turnover.items()}
    row.update(format_turnover)
    row["is_laundering"] = group["is_laundering"].max()
    return row
