#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=10g
#SBATCH -t 0 

set -x  # echo commands to stdout
set -e  # exit on error

source activate dynet
