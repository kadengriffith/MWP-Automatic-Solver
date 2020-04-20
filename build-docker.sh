#!/bin/bash

chmod a+x run.sh && chmod a+x trial.sh && chmod a+x make-data.sh

docker build --rm -t griffith/mwp-automatic-solver .

./make-data.sh