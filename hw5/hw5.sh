# bash hw5.sh [food dataset dir] [output image dir]
FOOD=$1
IMAGE=$2

EXPLAIN='./explainable.py'
DEEPDREAM='./deepdream.py'
MODEL='./cnn_weight'

wget 'https://github.com/jacky12123/ML2020/releases/download/PReLU%2BDropout/cnn_weight'
python3 $EXPLAIN $FOOD $MODEL $IMAGE
python3 $DEEPDREAM $FOOD $MODEL $IMAGE