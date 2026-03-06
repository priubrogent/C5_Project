#!/bin/bash
#SBATCH -n 4 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /hhome/priubrogent/mcvpol/C5/Week2/ # working directory
#SBATCH -p dcca40 # Partition to submit to
#SBATCH --mem 16000
#SBATCH -o %x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e %x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1

# python /hhome/priubrogent/mcvpol/C5/Week2/single_inference.py \
#     --image 0013/000105.png


python single_inference.py --image 0013/000105.png --strategy random_mask --num_points 3