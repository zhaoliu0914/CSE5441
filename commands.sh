ssh -X -C zhaoliu@owens.osc.edu

salloc -N <number_nodes> -A PAS2661 -pserial time=<time_in_minutes>

salloc -N <number_nodes> -A PAS2661 -p gpudebug --gpus-per-node=1 --time=<time_in_minutes>

salloc -N 1 -A PAS2661 -p gpuserial --gpus-per-node=1 --time=20


ulimit -a
ulimit -c 2048


module list
module load cuda/11.6.1

nvidia-smi

sbatch -A PAS2661 --time=30 batch_file_name


squeue -u <username>


salloc  -A sfaf -p serial

sinteractive

watch squeue -u 

scontrol show job 

job dependency -d