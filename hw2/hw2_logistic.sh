# bash ./hw2_logistic.sh $1 $2 $3 $4 $5 $6
X_TRAIN=$3
Y_TRAIN=$4
X_TEST=$5
OUTPUT=$6

LOGISTIC="./logistic/logistic.py"
python $LOGISTIC $X_TRAIN $Y_TRAIN $X_TEST $OUTPUT