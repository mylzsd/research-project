#!/bin/sh

for d in audiology breast_cancer breast_w cmc dematology ecoli glass hepatitis human_activity iris lymphography
do
	for n in 50 100
	do
		echo "Running python src/test.py -m mlp -d ${d} -n ${n} > out/mlp/mlp_${d}_${n}.out"
		python src/test.py -m mlp -d ${d} -n ${n} > out/mlp/mlp_${d}_${n}.out
	done
done
