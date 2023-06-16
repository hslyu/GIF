#!/bin/bash

for alpha in $(seq -f "%.2f" 0.00 0.01 0.25)
do
	if [ $# -eq 0 ]; then
		python scripts/train.py --num_epoch 50 --alpha $alpha
	elif [ $# -eq 1 ]; then
		python scripts/train.py --num_epoch 50 --alpha $alpha --loss $1
	elif [ $# -eq 2 ]; then
		python scripts/train.py --num_epoch 50 --alpha $alpha --loss $1 --data $2
	fi
    message="\"python scripts/train.py --num_epoch 50 --alpha $alpha\" has been finished"
    echo $message
done
