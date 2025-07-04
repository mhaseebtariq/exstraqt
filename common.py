import os
import pickle
import random
import shutil
import time
import uuid
from glob import glob
from math import ceil
from pathlib import Path

import pandas as pd
import psutil


MULTI_PROC_ROOT = Path("multiprocessing")
MULTI_PROC_INPUT = MULTI_PROC_ROOT / "input"
MULTI_PROC_OUTPUT = MULTI_PROC_ROOT / "output"
OUTPUT_EXT = ".pickle"
PROC_ARGS_SEPARATOR = "__0__"


def load_dump(loc):
    with open(loc, "rb") as f:
        return pickle.load(f)


def dump_object_for_proc(obj, is_input, pandas=False):
    folder = MULTI_PROC_OUTPUT
    if is_input:
        folder = MULTI_PROC_INPUT        
    ext = OUTPUT_EXT
    if pandas:
        ext = ".parquet"
    file_loc = folder / (str(uuid.uuid4()) + ext)
    if pandas:
        obj.to_parquet(file_loc)
        return file_loc
    with open(file_loc, "wb") as f:
        pickle.dump(obj, f)
    return file_loc


def reset_multi_proc_staging():
    shutil.rmtree(MULTI_PROC_ROOT, ignore_errors=True)
    os.makedirs(MULTI_PROC_INPUT)
    os.mkdir(MULTI_PROC_OUTPUT)

    
def create_workload_for_multi_proc(size, iterator, num_procs, *params, shuffle=False):
    if shuffle:
        random.shuffle(iterator)
    reset_multi_proc_staging()
    params_as_pickles = []
    for param in params:
        params_as_pickles.append(dump_object_for_proc(param, True))
    processed = 0
    iterator_chunk = []
    iterator_chunk_as_pickles = []
    chunk_size = ceil(size / num_procs)
    for item in iterator:
        processed += 1
        iterator_chunk.append(item)
        if processed == chunk_size:
            iterator_chunk_as_pickles.append(dump_object_for_proc(iterator_chunk, True))
            processed = 0
            iterator_chunk = []
    if iterator_chunk:
        iterator_chunk_as_pickles.append(dump_object_for_proc(iterator_chunk, True))
    return iterator_chunk_as_pickles, params_as_pickles


def collect_multi_proc_output(pandas=False):
    if pandas:
        return pd.read_parquet(MULTI_PROC_OUTPUT)
    files = glob(f"{MULTI_PROC_OUTPUT}{os.sep}*{OUTPUT_EXT}")
    output = []
    for fl in files:
        output.append(load_dump(fl))
    return [x for y in output for x in y]


def construct_arguments(*args):
    return PROC_ARGS_SEPARATOR.join([str(x) for x in args])


def load_arguments(args):
    return args.split(PROC_ARGS_SEPARATOR)


def get_processes(ids):
    processes = []
    for process in psutil.process_iter():
        cmdline = []
        try:
            cmdline = process.cmdline()
        except:  # noqa
            pass
        if ids.intersection(cmdline):
            processes.append(process)
    return processes


def wait_for_processes(process_ids):
    try:
        while get_processes(process_ids):
            time.sleep(5)
    except KeyboardInterrupt:
        for proc in get_processes(process_ids):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
    finally:
        for proc in get_processes(process_ids):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass


def get_weights(data_aggregated):
    source_totals = (
        data_aggregated.groupby("source")
        .agg({"amount": "sum"})["amount"]
        .to_dict()
    )
    target_totals = (
        data_aggregated.groupby("target")
        .agg({"amount": "sum"})["amount"]
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
            (x["amount"] / x["total_sent_by_source"])
            + (x["amount"] / x["total_received_by_target"])
        ),
        axis=1,
    )
    return data_aggregated.loc[:, ["source", "target", "weight"]]