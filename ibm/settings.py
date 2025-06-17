import os


TIMESTAMP_FORMAT = "yyyy/MM/dd HH:mm"
VALID_SIZES = ["Small", "Medium", "Large"]
HIGH_ILLICIT = True
FILE_SIZE = "Small"
assert FILE_SIZE in VALID_SIZES
ILLICIT_TYPE = "HI" if HIGH_ILLICIT else "LI"
MAIN_LOCATION = os.path.join(os.path.curdir, "synthetic-data")
STAGED_DATA_LOCATION = os.path.join(MAIN_LOCATION, "staged-transactions")
STAGED_CASES_DATA_LOCATION = os.path.join(
    MAIN_LOCATION, "staged-cases-transactions.parquet"
)
STAGED_DATA_CSV_LOCATION = os.path.join(MAIN_LOCATION, "staged-transactions.csv")
STAGED_PATTERNS_CSV_LOCATION = os.path.join(MAIN_LOCATION, "staged-patterns.txt")
DATA_FILE = os.path.join(MAIN_LOCATION, f"{ILLICIT_TYPE}-{FILE_SIZE}_Trans.csv")
PATTERNS_FILE = os.path.join(MAIN_LOCATION, f"{ILLICIT_TYPE}-{FILE_SIZE}_Patterns.txt")
