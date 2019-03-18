#!/bin/bash
# Job name:
#SBATCH --job-name=phil_nmep_hw_6
#
# Account:
#SBATCH --account=co_mlab
#
# Partition:
#SBATCH --partition=savio2_1080ti
#
# Wall clock limit:
#SBATCH --time=8:15:30
#
#SBATCH --cpus-per-task=2
#
#SBATCH --qos=mlab_1080ti2_normal
#
#SBATCH --nodes=1
#
#SBATCH --ntasks=1
#
#SBATCH --gres=gpu:1
#
## Command(s) to run:
module load tensorflow/1.12.0-py36-pip-gpu
python3 main.py \
--model_id=-1 \
--train=True \
--data_dir=data \
--config_yaml=config.yaml
echo "yeet"
