# timeout 300 bash hw6_fgsm.sh <input dir> <output img dir>
INPUT=$1
OUTPUT=$2

ATTACK='./fgsm/densenet121.py'
python3 $ATTACK $INPUT $OUTPUT