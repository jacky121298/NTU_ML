# bash hw3_train.sh <data directory>
DATA_DIR=$1
TRAIN='./cnn/train.py'
CNN_W='./cnn/cnn_weight'

python3 $TRAIN $DATA_DIR $CNN_W