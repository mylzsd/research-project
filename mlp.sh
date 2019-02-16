#!/bin/sh

for i in breast_cancer breast_w cmc dematology ecoli glass hepatitis human_activity iris lymphography
do
	echo "Running python src/test.py -m mlp -d ${i} > out/mlp_${i}_50.out"
	python src/test.py -m mlp -d ${i} > out/mlp_${i}_50.out
	echo "Running python src/test.py -m mlp -d ${i} -n 100 > out/mlp_${i}_100.out"
	python src/test.py -m mlp -d ${i} -n 100 > out/mlp_${i}_100.out
done
