# bash train_baseline.sh <trainX_npy> <checkpoint>
TRAIN_X=$1
CHECKPOINT=$2

TRAIN='./baseline/train.py'
python3 $TRAIN $TRAIN_X $CHECKPOINT