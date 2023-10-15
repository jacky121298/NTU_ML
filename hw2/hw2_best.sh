# bash ./hw2_best.sh $1 $2 $3 $4 $5 $6
X_TRAIN=$3
Y_TRAIN=$4
X_TEST=$5
OUTPUT=$6

BEST="./best/logistic.py"
NP_FILE="./best/feature.npz"

python $BEST $X_TRAIN $Y_TRAIN $X_TEST $OUTPUT $NP_FILE