#!/bin/bash

for filename in *.yaml; do
    python translator.py $filename
done
