import importlib
import sys

from common import load_arguments


if __name__ == "__main__":
    arguments = sys.argv
    module, func = arguments[2].split(".")
    args = load_arguments(arguments[3])
    func = getattr(importlib.import_module(module), func)
    func(*args)