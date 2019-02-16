#!/bin/sh

for i in audiology breast_cancer breast_w cmc dematology ecoli glass hepatitis human_activity iris lymphography
do
	echo "Running python src/test.py -m examine -d ${i} -n 10 > out/exam_${i}_10.out"
	python src/test.py -m examine -d ${i} -n 10 > out/exam_${i}_10.out
	echo "Running python src/test.py -m examine -d ${i} -n 10 -r > out/exam_${i}_10_r.out"
	python src/test.py -m examine -d ${i} -n 10 -r > out/exam_${i}_10_r.out
	echo "Running python src/test.py -m examine -d ${i} -n 20 > out/exam_${i}_20.out"
	python src/test.py -m examine -d ${i} -n 20 > out/exam_${i}_20.out
	echo "Running python src/test.py -m examine -d ${i} -n 20 -r > out/exam_${i}_20_r.out"
	python src/test.py -m examine -d ${i} -n 20 -r > out/exam_${i}_20_r.out
done
