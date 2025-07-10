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


FEATURE_TYPES = {
    "key": str,
    "num_source_or_target": np.uint16,
    "num_source_and_target": np.uint16,
    "num_source_only": np.uint16,
    "num_target_only": np.uint16,
    "num_transactions": np.uint16,
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
    # TODO: This can be made much faster!
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
    }

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

    exploded = pd.DataFrame(
        df["timestamps_amounts"].explode().tolist(), columns=["ts", "amount_usd"]
    )

    features_row["ts_range"] = exploded["ts"].max() - exploded["ts"].min()
    features_row["ts_std"] = exploded["ts"].std()
    features_row["ts_weighted_mean"] = np.average(exploded["ts"], weights=exploded["amount_usd"])
    features_row["ts_weighted_median"] = weighted_quantiles(
        exploded["ts"].values, weights=exploded["amount_usd"].values, quantiles=0.5, interpolate=True
    )
    features_row["ts_weighted_std"] = weighted_std(exploded["ts"], exploded["amount_usd"])

    if graph_features:
        graph = ig.Graph.DataFrame(df[["source", "target"]], use_vids=False, directed=True)
        features_row["assortativity_degree"] = graph.assortativity_degree(directed=True)
        features_row["assortativity_degree_ud"] = graph.assortativity_degree(directed=False)
        features_row["max_degree"] = max(graph.degree(mode="all"))
        features_row["max_degree_in"] = max(graph.degree(mode="in"))
        features_row["max_degree_out"] = max(graph.degree(mode="out"))
        features_row["diameter"] = graph.diameter(directed=True, unconn=True)

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
            pd.concat(df_comms, ignore_index=True).to_parquet(f"{MULTI_PROC_STAGING_LOCATION}{os.sep}{index + 1}.parquet")
            df_comms = []
    
    if len(df_comms) > 1:
        pd.concat(df_comms, ignore_index=True).to_parquet(f"{MULTI_PROC_STAGING_LOCATION}{os.sep}{index + 1}.parquet")
    
    del df_comms

    response = spark.read.parquet(
        str(MULTI_PROC_STAGING_LOCATION)
    ).groupby("key").applyInPandas(generate_features_udf_wrapper(True), schema=SCHEMA_FEAT_UDF).toPandas()
    
    return pd.DataFrame(response["features"].apply(json.loads).tolist()).astype(FEATURE_TYPES)
