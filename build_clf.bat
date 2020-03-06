@echo off

FOR %%D in (abalone audio bw cmc credit glass heart ha iris lymph) DO (
	echo "Running python src/run.py -a ptdqn -d %%D > out/d%%Dc1f1_1.out"
	python src/run.py -a ptdqn -d %%D > out/d%%Dc1f1_1.out
)
pause