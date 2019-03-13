#!/bin/sh

for d in audiology breast_cancer breast_w cmc dematology ecoli glass hepatitis human_activity iris lymphography
do
	for n in 20 50 100
	do
		echo "Running python src/test.py -m examine -d ${d} -n ${n} > out/examine/exam_${d}_${n}.out"
		python src/test.py -m examine -d ${d} -n ${n} > out/examine/exam_${d}_${n}.out
		echo "Running python src/test.py -m examine -d ${d} -n ${n} -u > out/examine/exam_${d}_${n}_u.out"
		python src/test.py -m examine -d ${d} -n ${n} -u > out/examine/exam_${d}_${n}_u.out
	done
done
