import os
import pickle
import random
import sys
import uuid

from common import load_dump, create_workload_for_multi_proc


THRESHOLD_RANK = 0.005


def get_top_n(sub_graph, queries):
    ranks = sub_graph.personalized_pagerank(
        reset_vertices=queries,
        directed=False,
        damping=0.95,
        weights="weight",
        implementation="prpack",
    )
    ranks = sorted(
        zip([x["name"] for x in sub_graph.vs()], ranks),
        reverse=True,
        key=lambda x: x[1],
    )
    return {x[0] for x in ranks if x[1] >= THRESHOLD_RANK}


def get_communities_chunk(args):
    nodes_loc, graph_loc = args
    graph_chunk = load_dump(graph_loc)
    communities_chunk = []
    for node, neighborhood in load_dump(nodes_loc):
        sub_g = graph_chunk.induced_subgraph(neighborhood)
        communities_chunk.append((node, get_top_n(sub_g, [node])))
    return communities_chunk


def get_communities_spark(nodes_neighborhoods, graph, num_procs, spark):
    nodes_neighborhoods = list(nodes_neighborhoods.items())
    nodes_locations, params = create_workload_for_multi_proc(
        len(nodes_neighborhoods), nodes_neighborhoods, num_procs, graph, shuffle=True
    )
    del nodes_neighborhoods
    graph_location = params[0]
    partitions = [(x, graph_location) for x in nodes_locations]
    return [
        x for y in spark.sparkContext.parallelize(partitions, num_procs).map(get_communities_chunk).collect()
        for x in y
    ]