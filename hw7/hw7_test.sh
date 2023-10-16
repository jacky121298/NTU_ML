# bash hw7_test.sh <data directory> <prediction file>
DATA_DIR=$1
SUBMIT=$2

TEST='./test.py'
MODEL='./model/8_bit_model.pkl'

python3 $TEST $DATA_DIR $MODEL 1 $SUBMIT