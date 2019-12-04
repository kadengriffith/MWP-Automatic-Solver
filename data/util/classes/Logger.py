from __future__ import absolute_import
import os

DIR_PATH = os.path.abspath(os.path.dirname(__file__))


class Logger():
    def __init__(self, model_name):
        self.model = model_name

    def log(self, what):
        # Append to the model's log file
        with open(os.path.join(DIR_PATH, f"../../../models/trained/{self.model}.txt"), 'a+') as fh:
            fh.write(what + '\n')

    def plog(self, what):
        # Print then log
        print(what)
        self.log(what)
