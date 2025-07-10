import os


TIMESTAMP_FORMAT = "yyyy/MM/dd HH:mm"
VALID_SIZES = ["Small", "Medium", "Large"]

HIGH_ILLICIT = True
FILE_SIZE = "Medium"

assert FILE_SIZE in VALID_SIZES
ILLICIT_TYPE = "HI" if HIGH_ILLICIT else "LI"
MAIN_LOCATION = os.path.join(os.path.curdir, "synthetic-data")
DATA_FILE = os.path.join(MAIN_LOCATION, f"{ILLICIT_TYPE}-{FILE_SIZE}_Trans.csv")
PATTERNS_FILE = os.path.join(MAIN_LOCATION, f"{ILLICIT_TYPE}-{FILE_SIZE}_Patterns.txt")

OUTPUT_POSTFIX = f"-{ILLICIT_TYPE.lower()}-{FILE_SIZE.lower()}"
STAGED_DATA_LOCATION = os.path.join(MAIN_LOCATION, f"staged-transactions{OUTPUT_POSTFIX}")
STAGED_CASES_DATA_LOCATION = os.path.join(
    MAIN_LOCATION, f"staged-cases-transactions{OUTPUT_POSTFIX}.parquet"
)
STAGED_DATA_CSV_LOCATION = os.path.join(MAIN_LOCATION, f"staged-transactions{OUTPUT_POSTFIX}.csv")
STAGED_PATTERNS_CSV_LOCATION = os.path.join(MAIN_LOCATION, f"staged-patterns{OUTPUT_POSTFIX}.txt")

G_FLOW_PREFIX = "graph_flow_feat_"
G_COMM_PREFIX = "graph_comm_feat_"
G_1HOP_PREFIX = "graph_1_hop_feat_"
G_GLOB_PREFIX = "graph_global_"

assert G_FLOW_PREFIX != G_COMM_PREFIX != G_1HOP_PREFIX != G_GLOB_PREFIX
