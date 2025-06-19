import os
import pickle
import sys
import uuid

from common import (
    load_dump, create_workload_for_multi_proc, wait_for_processes,
    dump_object_for_proc, construct_arguments, collect_multi_proc_output
)


THRESHOLD_RANK = 0.01
NEIGHBORS_ONLY = False


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


def get_communities_chunk(nodes_loc, graph_loc):
    nodes = load_dump(nodes_loc)
    graph = load_dump(graph_loc)
    communities = []
    for node in nodes:
        neighborhood = graph.neighborhood(node, order=2, mode="all", mindist=0)
        neighborhood = {x["name"] for x in graph.vs(neighborhood)}
        sub_g = graph.induced_subgraph(neighborhood)
        communities.append((node, get_top_n(sub_g, [node])))
    dump_object_for_proc(communities, False)


def get_communities_multi_proc(nodes, graph, num_procs):
    nodes_locations, params = create_workload_for_multi_proc(
        len(nodes), nodes, num_procs, graph, shuffle=True
    )
    graph_loc = params[0]
    proc = "communities.get_communities_chunk"
    process_ids = set()
    for nodes_location in nodes_locations:
        process_id = str(uuid.uuid4())
        args = construct_arguments(nodes_location, graph_loc)
        os.system(f"{sys.executable} spawn.py {process_id} {proc} {args} &")
        process_ids = process_ids.union([process_id])
    wait_for_processes(process_ids)
    return collect_multi_proc_output()