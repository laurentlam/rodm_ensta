#!/bin/bash

while getopts ":d:frs" arg; do
    case $arg in
        d) DATASET=$OPTARG; echo "DATASET_NAME: $DATASET";;
        f) echo "Creating features for $DATASET"; python3 src/pipeline.py -d $DATASET;;
        r) echo "Generating new rules for $DATASET"; rm -rf ./res/${DATASET}_rules.csv;;
        s) echo "Compute new order for rules of $DATASET"; rm -rf ./res/${DATASET}_ordered_rules.csv;;
        \?) echo "Example usage shown below: \n ./src/pipeline.sh -d DATASET_NAME -f True -r True -s True";;
    esac
done

julia src/main.jl $DATASET