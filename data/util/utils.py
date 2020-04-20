from tensorflow import logical_and, size
from re import sub
from pickle import load, dump

# Data constraints
MAX_LENGTH = 60


def load_data_from_binary(absolute_path):
    # Get the lines in a binary as list
    with open(absolute_path, "rb") as fh:
        file_data = load(fh)

    return file_data


def to_binary(absolute_path, what):
    # Save to a binary
    with open(absolute_path, 'wb') as fh:
        dump(what, fh)


def get_as_tuple(example):
    # Separate the trainable data
    ex_as_dict = dict(example)

    return ex_as_dict["question"], ex_as_dict["equation"]


def expressionize(what):
    # It may help training if the 'x =' is not learned
    what = sub(r"([a-z] \=|\= [a-z])", "", what)
    return sub(r"^\s+", "", what)


def print_epoch(what, clear=False):
    # Overwrite the line to see live updated results
    print(f"{what}", end="\r")

    if clear:
        # Clear the line being overwritten by print_epoch
        print()


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return logical_and(size(x) <= max_length,
                       size(y) <= max_length)
