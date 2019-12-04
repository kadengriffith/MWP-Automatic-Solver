from __future__ import absolute_import

import os
import re
import pickle

DIR_PATH = os.path.abspath(os.path.dirname(__file__))


def load_data_from_binary(absolute_path):
    # Get the lines in a binary as list
    with open(absolute_path, "rb") as fh:
        file_data = pickle.load(fh)

    return file_data


def to_binary(absolute_path, what):
    # Save to a binary
    with open(absolute_path, 'wb') as fh:
        pickle.dump(what, fh)


def get_as_tuple(example):
    # Separate the trainable data
    ex_as_dict = dict(example)

    return ex_as_dict["question"], ex_as_dict["equation"]


def expressionize(what):
    # It may help training if the 'x =' is not learned
    what = re.sub(r"([a-z] \=|\= [a-z])", "", what)
    return re.sub(r"^\s+", "", what)


def print_epoch(what, clear=False):
    # Overwrite the line to see live updated results
    print(f"{what}\r", end="")

    if clear:
        # Clear the line being overwritten by print_epoch
        print()
