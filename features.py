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

from common import reset_multi_proc_staging, MULTI_PROC_INPUT


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

FEATURE_TYPES = {
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
        "num_transactions": df["num_transactions"].sum(),
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
        turnover_currency_norm[key] = float((CURRENCY_RATES[key] * value) / (turnover or 1))

    features_row["turnover"] = turnover
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
            generate_features(df, row["key"], graph_features=graph_features),
            allow_nan=True, cls=NpEncoder,
        )
        return pd.DataFrame([{"features": features}])
    return generate_features_udf


def generate_features_spark(communities, graph, spark):
    reset_multi_proc_staging()
    chunk_size = 100_000
    
    df_comms = []
    for index, (node, comm) in enumerate(communities):
        df_comm = graph.induced_subgraph(comm).get_edge_dataframe()
        if not df_comm.empty:
            df_comm.loc[:, "key"] = node
            df_comms.append(df_comm)
        if not ((index + 1) % chunk_size):
            pd.concat(df_comms, ignore_index=True).to_parquet(f"{MULTI_PROC_INPUT}{os.sep}{index + 1}.parquet")
            df_comms = []
    
    if len(df_comms) > 1:
        pd.concat(df_comms, ignore_index=True).to_parquet(f"{MULTI_PROC_INPUT}{os.sep}{index + 1}.parquet")
    
    del df_comms

    response = spark.read.parquet(
        str(MULTI_PROC_INPUT)
    ).groupby("key").applyInPandas(generate_features_udf_wrapper(True), schema=SCHEMA_FEAT_UDF).toPandas()
    
    return pd.DataFrame(response["features"].apply(json.loads).tolist()).astype(FEATURE_TYPES)


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
    row = {"source": src, "target": tgt, "total_amount": total}
    row.update(currency_turnover)
    format_turnover = (
        df
        .groupby("format")
        .agg({"amount": "sum"})
    ).to_dict()["amount"]
    format_turnover = {k.lower().replace(" ", "_"): v / total for k, v in format_turnover.items()}
    row.update(format_turnover)
    
    return pd.DataFrame([json.dumps(row)], columns=["features"])
