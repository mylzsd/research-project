#!/bin/sh

for d in abalone audio bw cmc credit glass heart ha iris lymph
do
	echo "Running python src/run.py -a ptdqn -d ${d} > out/d${d}c1f1_2.out"
	python src/run.py -a ptdqn -d ${d} > out/d${d}c1f1_2.out
done
