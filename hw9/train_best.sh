# bash train_best.sh <trainX_npy> <checkpoint>
TRAIN_X=$1
CHECKPOINT=$2

TRAIN='./best/train.py'
python3 $TRAIN $TRAIN_X $CHECKPOINT