@echo off

FOR %%D in (audiology breast_cancer breast_w cmc dematology ecoli glass hepatitis human_activity iris lymphography) DO (
	FOR %%N in (50 100) DO (
		echo "Running python src/test.py -m mlp -d %%D -n %%N > out/mlp/mlp_%%D_%%N.out"
		python src/test.py -m mlp -d %%D -n %%N > out/mlp/mlp_%%D_%%N.out
	)
)
pause