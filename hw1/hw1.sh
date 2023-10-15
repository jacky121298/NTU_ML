# bash hw1.sh [input file] [output file]

INPUT_CSV=$1
OUTPUT_CSV=$2
#TRAIN_CSV="./data/train.csv"

#TRAIN_FILE="./simple/train.py"
TEST_FILE="./simple/test.py"
NP_FILE="./simple/weight.npz"

#python $TRAIN_FILE $TRAIN_CSV $NP_FILE
python $TEST_FILE $INPUT_CSV $NP_FILE $OUTPUT_CSV