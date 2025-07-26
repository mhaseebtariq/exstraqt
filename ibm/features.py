import json
import os
import pickle
import sys
import uuid

import numpy as np
import pandas as pd
import igraph as ig

from pyspark.sql import functions as sf
from pyspark.sql import types as st

from common import reset_multi_proc_staging, MULTI_PROC_STAGING_LOCATION


SCHEMA_FEAT_UDF = st.StructType([st.StructField("features", st.StringType())])

CURRENCY_RATES = {
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
    return sources, targets, source_or_target, source_and_target, source_only, target_only


def generate_features(df, row, graph_features=False):
    # TODO: This can be made much faster!
    sources, targets, source_or_target, source_and_target, source_only, target_only = get_segments(
        "source", "target", df
    )
    node_name = row["key"]
    features_row = {
        "key": node_name,
        "num_sources": len(sources),
        "num_targets": len(targets),
        "num_source_or_target": len(source_or_target),
        "num_source_and_target": len(source_and_target),
        "num_source_only": len(source_only),
        "num_target_only": len(target_only),
        "num_transactions": df["num_transactions"].sum(),
        "num_currencies": df["source_currency"].nunique(),
    }

    sources, targets, source_or_target, source_and_target, source_only, target_only = get_segments(
        "source_bank", "target_bank", df
    )
    features_row["num_source_banks"] = len(sources)
    features_row["num_target_banks"] = len(targets)
    features_row["num_source_or_target_bank"] = len(source_or_target)
    features_row["num_source_and_target_bank"] = len(source_and_target)
    features_row["num_source_only_bank"] = len(source_only)
    features_row["num_target_only_bank"] = len(target_only)

    left = (
        df.loc[:, ["target", "source_currency", "amount"]]
        .rename(columns={"target": "source"})
        .groupby(["source", "source_currency"])
        .agg({"amount": "sum"})
    )
    right = df.groupby(["source", "source_currency"]).agg({"amount": "sum"})
    result = left.join(right, how="outer", lsuffix="_left").fillna(0).reset_index()
    result.loc[:, "delta"] = result["amount_left"] - result["amount"]
    turnover_currency = result[result["delta"] > 0].reset_index(drop=True)
    turnover_currency = (
        turnover_currency.groupby("source_currency").agg({"delta": "sum"}).to_dict()["delta"]
    )

    agg = {"amount": "sum", "amount_weighted": "sum"}
    columns = ["amount", "amount_weighted"]
    left = df.loc[:, ["target"] + columns].rename(columns={"target": "source"}).groupby("source").agg(agg)
    features_row["max_credit_edges"] = np.max(left["amount"])
    features_row["mean_credit_edges"] = np.mean(left["amount"])
    features_row["median_credit_edges"] = np.median(left["amount"])
    features_row["std_credit_edges"] = np.std(left["amount"])
    features_row["max_credit_edges_weighted"] = np.max(left["amount_weighted"])
    features_row["mean_credit_edges_weighted"] = np.mean(left["amount_weighted"])
    features_row["median_credit_edges_weighted"] = np.median(left["amount_weighted"])
    features_row["std_credit_edges_weighted"] = np.std(left["amount_weighted"])
    
    right = df.loc[:, ["source"] + columns].groupby("source").agg(agg)
    features_row["max_debit_edges"] = np.max(right["amount"])
    features_row["mean_debit_edges"] = np.mean(right["amount"])
    features_row["median_debit_edges"] = np.median(right["amount"])
    features_row["std_debit_edges"] = np.std(right["amount"])
    features_row["max_debit_edges_weighted"] = np.max(right["amount_weighted"])
    features_row["mean_debit_edges_weighted"] = np.mean(right["amount_weighted"])
    features_row["median_debit_edges_weighted"] = np.median(right["amount_weighted"])
    features_row["std_debit_edges_weighted"] = np.std(right["amount_weighted"])
    
    result = left.join(right, how="outer", lsuffix="_left").fillna(0).reset_index()
    result.loc[:, "delta"] = result["amount_left"] - result["amount"]
    turnover = float(result[result["delta"] > 0]["delta"].sum())
    features_row["turnover"] = turnover

    turnover_currency_norm = {}
    for key, value in turnover_currency.items():
        turnover_currency_norm[key] = float((CURRENCY_RATES[key] * value) / turnover)

    features_row.update(turnover_currency_norm)

    exploded = pd.DataFrame(
        df["timestamps_amounts"].explode().tolist(), columns=["ts", "amount"]
    )
    features_row["ts_range"] = exploded["ts"].max() - exploded["ts"].min()
    features_row["ts_std"] = exploded["ts"].std()
    features_row["ts_weighted_mean"] = np.average(exploded["ts"], weights=exploded["amount"])
    features_row["ts_weighted_median"] = weighted_quantiles(
        exploded["ts"].values, weights=exploded["amount"].values, quantiles=0.5, interpolate=True
    )
    features_row["ts_weighted_std"] = weighted_std(exploded["ts"], exploded["amount"])

    if graph_features:
        graph = ig.Graph.DataFrame(df[["source", "target"]], use_vids=False, directed=True)
        features_row["assortativity_degree"]= graph.assortativity_degree(directed=True)
        features_row["assortativity_degree_ud"] = graph.assortativity_degree(directed=False)
        features_row["max_degree"] = max(graph.degree(mode="all"))
        features_row["max_degree_in"] = max(graph.degree(mode="in"))
        features_row["max_degree_out"] = max(graph.degree(mode="out"))
        features_row["diameter"] = graph.diameter(directed=True, unconn=True)
        features_row["diameter_ud"] = graph.diameter(directed=False, unconn=True)
        features_row["density"] = graph.density(loops=False)
        biconn_components, articulation_points = graph.biconnected_components(return_articulation_points=True)
        features_row["num_biconn_components"] = len(biconn_components) 
        features_row["num_articulation_points"] = len(articulation_points)
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


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def generate_features_udf_wrapper(graph_features):
    def generate_features_udf(df):
        row = df.iloc[0]
        features = json.dumps(
            generate_features(df, row, graph_features=graph_features),
            allow_nan=True, cls=NpEncoder,
        )
        return pd.DataFrame([{"features": features}])
    return generate_features_udf


def generate_features_spark(communities, graph, spark):
    reset_multi_proc_staging()
    chunk_size = 100_000
    
    df_comms = []
    partitions = 0
    for index, (node, comm) in enumerate(communities):
        sub_g = graph.induced_subgraph(comm)
        df_comm = sub_g.get_edge_dataframe()
        if not df_comm.empty:
            df_comm.loc[:, "key"] = node
            df_comms.append(df_comm)
        if not ((index + 1) % chunk_size):
            partitions += 1
            pd.concat(df_comms, ignore_index=True).to_parquet(f"{MULTI_PROC_STAGING_LOCATION}{os.sep}{index + 1}.parquet")
            df_comms = []
    
    if len(df_comms) > 0:
        partitions += 1
        pd.concat(df_comms, ignore_index=True).to_parquet(f"{MULTI_PROC_STAGING_LOCATION}{os.sep}{index + 1}.parquet")
    
    del df_comms

    partitions *= 6
    partitions = (partitions % os.cpu_count()) + partitions

    response = spark.read.parquet(
        str(MULTI_PROC_STAGING_LOCATION)
    ).repartition(int(partitions), "key").groupby("key").applyInPandas(
        generate_features_udf_wrapper(True), schema=SCHEMA_FEAT_UDF
    ).toPandas()
    
    return pd.DataFrame(response["features"].apply(json.loads).tolist())


def get_edge_features_udf(df):
    row = df.iloc[0]
    src, tgt = row["source"], row["target"]
    
    currency_turnover = (
        df
        .groupby("source_currency")
        .agg({"source_amount": "sum"})
    ).to_dict()["source_amount"]
    total = df["amount"].sum()
    currency_turnover = {k: v / total for k, v in currency_turnover.items()}
    row = {
        "source": src, "target": tgt, 
        "total_amount": total, 
        "related_for": df["max_ts"].max() - df["min_ts"].min()
    }
    row.update(currency_turnover)
    format_turnover = (
        df
        .groupby("format")
        .agg({"amount": "sum"})
    ).to_dict()["amount"]
    format_turnover = {k.lower().replace(" ", "_"): v / total for k, v in format_turnover.items()}
    row.update(format_turnover)
    
    return pd.DataFrame([json.dumps(row, allow_nan=True, cls=NpEncoder)], columns=["features"])
