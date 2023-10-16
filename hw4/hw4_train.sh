# bash hw4_train.sh <training label data> <training unlabel data>
LABEL=$1
UNLABEL=$2

TRAIN='./semi/train.py'
W2V='./semi/model/w2v.model'

python3 $TRAIN $LABEL $UNLABEL $W2V