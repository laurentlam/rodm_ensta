
DATASET=$1

python3 src/pipeline.py -d $DATASET
julia src/main.jl $DATASET