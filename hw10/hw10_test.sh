# bash hw10_test.sh <test.npy> <model> <prediction.csv>
TEST_PATH=$1
MODEL=$2
SUBMIT=$3

if [ `echo $MODEL | grep -c "baseline" ` -gt 0 ]; then
    TEST='./baseline/test.py'
    TYPE='cnn'
else
    TEST='./best/test.py'
    TYPE='fcn'
fi

python3 $TEST $TEST_PATH $TYPE $MODEL $SUBMIT