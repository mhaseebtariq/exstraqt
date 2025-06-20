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

types = {
    "key": str,
    "num_source_or_target": np.uint16,
    "num_source_and_target": np.uint16,
    "num_source_only": np.uint16,
    "num_target_only": np.uint16,
    "num_timestamp_trx": np.uint16,
    "zero_trx": np.uint16,
    "non_zero_trx": np.uint16,
    "total_trx": np.uint16,
    "perc_zero_trx": np.float64,
    "perc_timestamp_trx": np.float64,
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
    "mean_degree": np.float16,
    "mean_degree_in": np.float16,
    "mean_degree_out": np.float16,
    "diameter": np.uint8,
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
        "num_timestamp_trx": df.shape[0],
    }
    zero_trx = df[df["is_zero_transaction"]]["num_transactions"].sum()
    non_zero_trx = df[~df["is_zero_transaction"]]["num_transactions"].sum()
    features_row["zero_trx"] = zero_trx
    features_row["non_zero_trx"] = non_zero_trx
    features_row["total_trx"] = zero_trx + non_zero_trx
    features_row["perc_zero_trx"] = zero_trx / features_row["total_trx"]
    features_row["perc_timestamp_trx"] = features_row["num_timestamp_trx"] / features_row["total_trx"]
    
    left = (
        df.loc[:, ["target", "amount_usd"]]
        .rename(columns={"target": "source"})
        .groupby("source")
        .agg({"amount_usd": "sum"})
    )
    right = df.groupby("source").agg({"amount_usd": "sum"})
    result = left.join(right, how="outer", lsuffix="_left").fillna(0).reset_index()
    result.loc[:, "delta"] = result["amount_usd_left"] - result["amount_usd"]
    turnover = float(result[result["delta"] > 0]["delta"].sum())

    features_row["turnover"] = turnover

    features_row["ts_range"] = df["window_delta"].max() - df["window_delta"].min()
    features_row["ts_std"] = df["window_delta"].std()
    features_row["ts_weighted_mean"] = np.average(df["window_delta"], weights=df["amount_usd"])
    features_row["ts_weighted_median"] = weighted_quantiles(
        df["window_delta"].values, weights=df["amount_usd"].values, quantiles=0.5, interpolate=True
    )
    features_row["ts_weighted_std"] = weighted_std(df["window_delta"], df["amount_usd"])

    if graph_features:
        graph = ig.Graph.DataFrame(df[["source", "target"]], use_vids=False, directed=True)
        features_row["assortativity_degree"] = graph.assortativity_degree(directed=True)
        features_row["assortativity_degree_ud"] = graph.assortativity_degree(directed=False)
        degree_all = graph.degree(mode="all")
        degree_in = graph.degree(mode="in")
        degree_out = graph.degree(mode="out")
        features_row["max_degree"] = max(degree_all)
        features_row["max_degree_in"] = max(degree_in)
        features_row["max_degree_out"] = max(degree_out)
        features_row["mean_degree"] = np.mean(degree_all)
        features_row["mean_degree_in"] = np.mean(degree_in)
        features_row["mean_degree_out"] = np.mean(degree_out)
        features_row["diameter"] = graph.diameter(directed=True, unconn=True)

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
    total = group["amount_usd"].sum()
    row = {"source": src, "target": tgt, "total_amount_usd": total, "count_trx": len(group)}
    ts = (group["timestamp"].astype(int) / 10**9).values
    row["ts_range"] = ts.max() - ts.min()
    row["ts_std"] = ts.std()
    return row
