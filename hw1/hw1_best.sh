# bash hw1_best.sh [input file] [output file]

INPUT_CSV=$1
OUTPUT_CSV=$2
#TRAIN_CSV="./data/train.csv"

#TRAIN_FILE="./best/train.py"
TEST_FILE="./best/test.py"
NP_FILE="./best/weight.npz"

#python $TRAIN_FILE $TRAIN_CSV $NP_FILE
python $TEST_FILE $INPUT_CSV $NP_FILE $OUTPUT_CSV