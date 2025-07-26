import os
import pickle
import random
import sys
import uuid

from common import load_dump, create_workload_for_multi_proc


def get_top_n(sub_graph, query, threshold, weight_column):
    names = [x["name"] for x in sub_graph.vs()]
    query_indexes = [names.index(x) for x in query]
    ranks = sub_graph.personalized_pagerank(
        reset_vertices=query,
        directed=False,
        damping=0.95,
        weights=weight_column,
        implementation="prpack",
    )
    ranks_zipped = zip([x["name"] for x in sub_graph.vs()], tuple(ranks))
    for x in query_indexes:
        del ranks[x]
    if not ranks:
        return set(query)
    max_rank = max(ranks)
    return {x[0] for x in ranks_zipped if (x[1] / max_rank) >= threshold} | set(query)


def get_communities_chunk(args):
    queries_loc, graph_loc, order, mode, threshold, weight_column = args
    graph_chunk = load_dump(graph_loc)
    communities_chunk = []
    for node, neighborhood in load_dump(queries_loc):
        if order == -1:
            sub_g = graph_chunk
        else:
            # neighborhood = graph_chunk.neighborhood(node, order=order, mode=mode, mindist=0)
            sub_g = graph_chunk.induced_subgraph(neighborhood)
        communities_chunk.append((node, get_top_n(sub_g, [node], threshold, weight_column)))
    return communities_chunk


def get_communities_spark(queries, graph, num_procs, spark, order, mode, threshold, weight_column):
    queries = list(queries.items())
    queries_locs, params = create_workload_for_multi_proc(len(queries), queries, num_procs, graph, shuffle=True)
    del queries
    graph_location = params[0]
    partitions = [(x, graph_location, order, mode, threshold, weight_column) for x in queries_locs]
    return [
        x for y in spark.sparkContext.parallelize(partitions, num_procs).map(get_communities_chunk).collect()
        for x in y
    ]
