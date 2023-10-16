# bash hw9_best.sh <trainX_npy> <checkpoint> <prediction_path>
TRAIN_X=$1
CHECKPOINT=$2
SUBMIT=$3

TEST='./best/test.py'
python3 $TEST $TRAIN_X $CHECKPOINT $SUBMIT