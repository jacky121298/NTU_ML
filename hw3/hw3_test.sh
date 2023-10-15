# bash hw3_test.sh <data directory> <prediction file>
DATA_DIR=$1
OUTPUT_CSV=$2
TEST='./cnn/test.py'
CNN_W='./cnn_weight'

wget 'https://github.com/jacky12123/ML2020/releases/download/PReLU%2BDropout/cnn_weight'
python3 $TEST $DATA_DIR $CNN_W $OUTPUT_CSV