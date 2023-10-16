# bash hw12_test.sh <data directory> <prediction file>
DATA=$1
SUBMIT=$2

TEST='./best/test.py'
EXTRACTOR='./best/model/extractor_model.bin'
PREDICTOR='./best/model/predictor_model.bin'

python3 $TEST $DATA $EXTRACTOR $PREDICTOR $SUBMIT