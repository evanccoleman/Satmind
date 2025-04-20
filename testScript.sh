#/bin/bash
#

# Getting to the right conda environment
# This activates the conda environment for the shell spawned 
# inside of the bash environment and assumes there's an
# applicable conda environment called 'SatmindNew'
eval "$(conda shell.bash hook)"
conda activate SatmindNew

# Need to add the parent (project) directory to the Python Path
# '-s' allows console output to the screen
PYTHONPATH=$(pwd) pytest tests/test_rl.py -s

