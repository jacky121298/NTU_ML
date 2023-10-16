# bash hw4_test.sh <testing data> <prediction file>
TESTING=$1
OUTPUT_CSV=$2

TEST='./semi/test.py'
W2V='./w2v.model'
MODEL='./ckpt.model'

wget 'https://github.com/jacky12123/ML2020/releases/download/w2v/w2v.model'
wget 'https://github.com/jacky12123/ML2020/releases/download/RNN/ckpt.model'
python3 $TEST $TESTING $W2V $MODEL $OUTPUT_CSV