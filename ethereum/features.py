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

from common import reset_multi_proc_staging, load_dump, create_workload_for_multi_proc, MULTI_PROC_STAGING_LOCATION


SCHEMA_FEAT_UDF = st.StructType([st.StructField("features", st.StringType())])


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
    }

    agg = {"amount_usd": "sum", "amount_usd_weighted": "sum"}
    columns = ["amount_usd", "amount_usd_weighted"]
    left = df.loc[:, ["target"] + columns].rename(columns={"target": "source"}).groupby("source").agg(agg)
    features_row["max_credit_edges"] = np.max(left["amount_usd"])
    features_row["mean_credit_edges"] = np.mean(left["amount_usd"])
    features_row["median_credit_edges"] = np.median(left["amount_usd"])
    features_row["std_credit_edges"] = np.std(left["amount_usd"])
    features_row["max_credit_edges_weighted"] = np.max(left["amount_usd_weighted"])
    features_row["mean_credit_edges_weighted"] = np.mean(left["amount_usd_weighted"])
    features_row["median_credit_edges_weighted"] = np.median(left["amount_usd_weighted"])
    features_row["std_credit_edges_weighted"] = np.std(left["amount_usd_weighted"])
    
    right = df.loc[:, ["source"] + columns].groupby("source").agg(agg)
    features_row["max_debit_edges"] = np.max(right["amount_usd"])
    features_row["mean_debit_edges"] = np.mean(right["amount_usd"])
    features_row["median_debit_edges"] = np.median(right["amount_usd"])
    features_row["std_debit_edges"] = np.std(right["amount_usd"])
    features_row["max_debit_edges_weighted"] = np.max(right["amount_usd_weighted"])
    features_row["mean_debit_edges_weighted"] = np.mean(right["amount_usd_weighted"])
    features_row["median_debit_edges_weighted"] = np.median(right["amount_usd_weighted"])
    features_row["std_debit_edges_weighted"] = np.std(right["amount_usd_weighted"])
    
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


def save_comm_transactions(args):
    graph_loc, comms_loc = args
    graph = load_dump(graph_loc)
    df_comms = []
    for node, comm in load_dump(comms_loc):
        sub_g = graph.induced_subgraph(comm)
        df_comm = sub_g.get_edge_dataframe()
        if not df_comm.empty:
            df_comm.loc[:, "key"] = node
            df_comms.append(df_comm.copy(deep=True))
    if df_comms:
        pd.concat(
            df_comms, ignore_index=True
        ).to_parquet(f"{MULTI_PROC_STAGING_LOCATION}{os.sep}{uuid.uuid4()}.parquet")


def generate_features_udf_wrapper(graph_features):
    def generate_features_udf(df):
        row = df.iloc[0]
        features = json.dumps(
            generate_features(df, row, graph_features=graph_features),
            allow_nan=True, cls=NpEncoder,
        )
        return pd.DataFrame([{"features": features}])
    return generate_features_udf


def save_comm_transactions(args):
    graph_loc, comms_loc = args
    graph = load_dump(graph_loc)
    df_comms = []
    for node, comm in load_dump(comms_loc):
        sub_g = graph.induced_subgraph(comm)
        df_comm = sub_g.get_edge_dataframe()
        if not df_comm.empty:
            df_vert = sub_g.get_vertex_dataframe()
            df_comm.loc[:, "src"] = df_comm["source"].apply(lambda x: df_vert.loc[x, "name"])
            df_comm.loc[:, "tgt"] = df_comm["target"].apply(lambda x: df_vert.loc[x, "name"])
            del df_comm["source"]
            del df_comm["target"]
            df_comm = df_comm.rename(columns={"src": "source", "tgt": "target"})
            df_comm.loc[:, "key"] = node
            df_comms.append(df_comm.copy(deep=True))
    if df_comms:
        pd.concat(
            df_comms, ignore_index=True
        ).to_parquet(f"{MULTI_PROC_STAGING_LOCATION}{os.sep}{uuid.uuid4()}.parquet")


def generate_features_spark(communities, graph_data, spark, num_cores=os.cpu_count()):
    reset_multi_proc_staging()
    chunk_size = 100_000

    graph = ig.Graph.DataFrame(graph_data.loc[:, ["source", "target"]], use_vids=False, directed=True)

    num_procs = int((np.floor((len(communities) / chunk_size) / num_cores) + 1) * num_cores)
    comms_locs, params = create_workload_for_multi_proc(len(communities), communities, num_procs, graph, shuffle=True)

    del graph
    del communities
    
    comms_partitions = [(params[0], x) for x in comms_locs]

    spark.sparkContext.parallelize(comms_partitions, len(comms_partitions)).map(save_comm_transactions).collect()

    for temp_loc in comms_locs + params:
        os.remove(temp_loc)

    graph_data = spark.createDataFrame(graph_data).withColumnRenamed("source", "src").withColumnRenamed("target", "trg")
    community_transactions = spark.read.parquet(str(MULTI_PROC_STAGING_LOCATION))
    community_transactions = community_transactions.join(
        graph_data,
        (community_transactions["source"] == graph_data["src"]) &
        (community_transactions["target"] == graph_data["trg"]),
        how="left"
    ).drop("src", "trg")

    response = community_transactions.groupby("key").applyInPandas(
        generate_features_udf_wrapper(True), schema=SCHEMA_FEAT_UDF
    ).toPandas()
    
    return pd.DataFrame(response["features"].apply(json.loads).tolist())
