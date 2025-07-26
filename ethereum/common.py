import gc
import os
import pickle
import random
import shutil
import sys
import uuid
from math import ceil
from pathlib import Path

import pandas as pd
import psutil


MULTI_PROC_ROOT = Path("multiprocessing")
MULTI_PROC_STAGING_LOCATION = MULTI_PROC_ROOT / os.environ["EXT_DATA_TYPE_FOLDER"]
OUTPUT_EXT = ".pickle"
PROC_ARGS_SEPARATOR = "__0__"


def load_dump(loc):
    with open(loc, "rb") as f:
        return pickle.load(f)


def dump_object_for_proc(obj, pandas=False):
    ext = OUTPUT_EXT
    if pandas:
        ext = ".parquet"
    file_loc = MULTI_PROC_STAGING_LOCATION / (str(uuid.uuid4()) + ext)
    if pandas:
        obj.to_parquet(file_loc)
        return file_loc
    with open(file_loc, "wb") as f:
        pickle.dump(obj, f)
    return file_loc


def reset_multi_proc_staging():
    shutil.rmtree(MULTI_PROC_STAGING_LOCATION, ignore_errors=True)
    os.makedirs(MULTI_PROC_STAGING_LOCATION)

    
def create_workload_for_multi_proc(size, iterator, num_procs, *params, shuffle=False):
    if shuffle:
        random.shuffle(iterator)
    reset_multi_proc_staging()
    params_as_pickles = []
    for param in params:
        params_as_pickles.append(dump_object_for_proc(param))
    processed = 0
    iterator_chunk = []
    iterator_chunk_as_pickles = []
    chunk_size = ceil(size / num_procs)
    for item in iterator:
        processed += 1
        iterator_chunk.append(item)
        if processed == chunk_size:
            iterator_chunk_as_pickles.append(dump_object_for_proc(iterator_chunk))
            processed = 0
            iterator_chunk = []
    if iterator_chunk:
        iterator_chunk_as_pickles.append(dump_object_for_proc(iterator_chunk))
    return iterator_chunk_as_pickles, params_as_pickles


def get_weights(data_aggregated):
    data_aggregated = data_aggregated.copy(deep=True)
    source_totals = (
        data_aggregated.groupby("source")
        .agg({"amount_usd": "sum"})["amount_usd"]
        .to_dict()
    )
    target_totals = (
        data_aggregated.groupby("target")
        .agg({"amount_usd": "sum"})["amount_usd"]
        .to_dict()
    )

    data_aggregated.loc[:, "total_sent_by_source"] = data_aggregated.loc[
        :, "source"
    ].apply(lambda x: source_totals[x])
    data_aggregated.loc[:, "total_received_by_target"] = data_aggregated.loc[
        :, "target"
    ].apply(lambda x: target_totals[x])
    data_aggregated.loc[:, "weight"] = data_aggregated.apply(
        lambda x: (
            (x["amount_usd"] / x["total_sent_by_source"])
            + (x["amount_usd"] / x["total_received_by_target"])
        ),
        axis=1,
    )
    return data_aggregated.loc[:, ["source", "target", "weight"]]


def delete_large_vars(globals_ref, locals_ref, max_size_in_mb=1):
    _ = gc.collect()
    for key in list(globals_ref.keys()):
        if isinstance(globals_ref[key], pd.DataFrame):
            del globals_ref[key]
            print(f"Deleted `global` DataFrame: {key}")
        elif (sys.getsizeof(globals_ref[key]) / 1024**2) > max_size_in_mb:
            del globals_ref[key]
            print(f"Deleted `global` large object: {key}")
    for key in list(locals_ref.keys()):
        if isinstance(locals_ref[key], pd.DataFrame):
            del locals_ref[key]
            print(f"Deleted `local` DataFrame: {key}")
        elif (sys.getsizeof(locals_ref) / 1024**2) > max_size_in_mb:
            del locals_ref[key]
            print(f"Deleted `local` large object: {key}")
    _ = gc.collect()
    return True
