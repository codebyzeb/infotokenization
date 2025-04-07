#!/bin/bash -l

# SLURM configuration
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --gres=gpu:4
#SBATCH --time=0-06:00:00
#SBATCH --account VLACHOS-SL2-GPU
#SBATCH --partition ampere
#SBATCH --signal=SIGUSR1@120
#SBATCH --job-name=1B  # <- change model here
#SBATCH --mail-type=ALL
#SBATCH --output=/rds/user/pl487/hpc-work/rdd/outputs/exp_1B/slurm_%j.out  # <- change model here
#SBATCH --error=/rds/user/pl487/hpc-work/rdd/outputs/exp_1B/slurm_%j.err   # <- change model here
#SBATCH --open-mode=append
#SBATCH --exclusive

model=1B # <- change model here

# Debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Environment setup
echo -e 'loading other modules'
. /etc/profile.d/modules.sh  # Leave this line (enables the module command)
module list
module purge  # Removes all modules still loaded
module load rhel8/default-amp
module load cuda/12.1
module load cudnn/8.9_cuda-12.1
module list

source .venv/bin/activate
uv sync

# Run script
srun uv run scripts/model_train_slurm.py \
    +callbacks.grad_accum.scheduling="{0: 2}" \
    data.batch_size=16 \
    data.eval_batch_size=128 \
    tok_name=bpe32000minipile \
    model=smol_llama-$model \
    pwd=/home/pl487/rds/hpc-work/rdd \
    torch_compile=true \
    trainer.devices=4 \
    data.num_workers=12 \
    hydra.run.dir=outputs/exp_$model \
    run_folder=. \
    resume_from_checkpoint=.checkpoints/last.ckpt
