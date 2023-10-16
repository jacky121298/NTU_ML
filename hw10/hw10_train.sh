# bash hw10_train.sh <train.npy> <model>
TRAIN_PATH=$1
MODEL=$2

if [ `echo $MODEL | grep -c "baseline" ` -gt 0 ]; then
    TRAIN='./baseline/train.py'
    TYPE='cnn'
else
    TRAIN='./best/train.py'
    TYPE='fcn'
fi

python3 $TRAIN $TRAIN_PATH $TYPE $MODEL