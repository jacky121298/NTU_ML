# bash train_improved.sh <trainX_npy> <checkpoint>
TRAIN_X=$1
CHECKPOINT=$2

TRAIN='./improved/train.py'
python3 $TRAIN $TRAIN_X $CHECKPOINT