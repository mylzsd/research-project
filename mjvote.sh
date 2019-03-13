#!/bin/sh

for r in 405 6676 5469 2305 3347
do
	echo "python src/test.py -m mjvote -r ${r} > out/mjvote/majorityVote_${r}.out"
	python src/test.py -m mjvote -r ${r} > out/mjvote/majorityVote_${r}.out
done
