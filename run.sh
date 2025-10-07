#!/bin/bash

#SBATCH --job-name=my_awesome_project
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=28G
#SBATCH --time 2-00:00:00
#SBATCH --partition batch
#SBATCH --output logs/%x_%j.out

~/workspace/miniconda3/bin/conda init
conda activate virtualenv
which python

cd /absolute/path/to/your/project/folder
python main.py --options1 tmp --options2 tmp