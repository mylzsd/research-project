#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition gpgpu

# Load required modules
module load Python/3.5.2-goolf-2015a

# Run codes
echo "Start Running"
time mpiexec -n 8 python3 src/driver.py -d audiology -m ql -n 50