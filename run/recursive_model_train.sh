#!/bin/bash

for path in $(seq 0 100);
do
	for alpha in $(seq -f "%.2f" 0.00 0.01 0.25)
	do
		python scripts/Figure_2-train.py --num_epoch 50 --alpha $alpha --path $path --network $1 --criterion $2 --data $3
		message="\"python scripts/train.py --num_epoch 50 --alpha $alpha\" has been finished"
		echo $message
	done
done
