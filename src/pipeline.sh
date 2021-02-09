
DATASET=$1

python pipeline.py -d $DATASET
julia main.jl $DATASET