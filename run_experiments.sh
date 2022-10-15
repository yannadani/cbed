EXPERIMENTS=100
SEEDS=$(seq $EXPERIMENTS)
for SEED in $SEEDS; do
    RANDOM=$SEED
    DATA_SEED=$RANDOM
    echo
    echo Running seed $DATA_SEED
    python experimental_design.py --data_seed $DATA_SEED "$@";
done
