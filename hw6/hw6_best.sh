# timeout 300 bash hw6_best.sh <input dir> <output img dir>
INPUT=$1
OUTPUT=$2

ATTACK='./best/best.py'
python3 $ATTACK $INPUT $OUTPUT