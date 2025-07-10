import os

MAIN_DATA_FILE = os.path.join(os.path.curdir, "data")
INPUT_GRAPH_FILE = os.path.join(MAIN_DATA_FILE, "MulDiGraph.pkl")
INPUT_DATA_FILE = os.path.join(MAIN_DATA_FILE, "data.parquet")
INPUT_DATA_ORIG_FILE = os.path.join(MAIN_DATA_FILE, "data_orig.parquet")
INPUT_RATES_FILE = os.path.join(MAIN_DATA_FILE, "rates.csv")

G_FLOW_PREFIX = "graph_flow_feat_"
G_COMM_PREFIX = "graph_comm_feat_"
G_1HOP_PREFIX = "graph_1_hop_feat_"
G_GLOB_PREFIX = "graph_global_"

assert G_FLOW_PREFIX != G_COMM_PREFIX != G_1HOP_PREFIX != G_GLOB_PREFIX
