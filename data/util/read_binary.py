from __future__ import absolute_import

import os
import sys
import pickle

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

# Simple command line tool to read the pickle files


def read_data_from_file(path):
    with open(path, "rb") as fh:
        file_data = pickle.load(fh)

    if len(sys.argv) > 2:
        limit = int(sys.argv[2])
    else:
        limit = 25
    return file_data[:limit]


if __name__ == "__main__":
    print(read_data_from_file(os.path.join(DIR_PATH,
                                           f"../{sys.argv[1]}")))
