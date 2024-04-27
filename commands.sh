ssh -X -C zhaoliu@owens.osc.edu

salloc -N <number_nodes> -A PAS2661 -pserial time=<time_in_minutes>

salloc -N <number_nodes> -A PAS2661 -p gpudebug --gpus-per-node=1 --time=<time_in_minutes>

salloc -N 1 -A PAS2661 -p gpuserial --gpus-per-node=1 --time=20

salloc -N 1 -A PAS2661 -p gpudebug --gpus-per-node=1 --time=60

salloc -N 5 --ntasks-per-node=5 -A PAS2661 -p parallel --time=30

srun -N 1 -n 2 -A PAS2661 -p debug --time=2 ./a.out

srun -N 5 -n 5 -A PAS2661 -p parallel --time=20 ./a.out

sbatch -N 5 -n 5 -A PAS2661 -p parallel --time=30 ./a.out


nvidia-smi

module load cuda
module load cuda/11.6.1


ulimit -a
ulimit -c 2048


module list


nvidia-smi

sbatch -A PAS2661 --time=30 batch_file_name


squeue -u <username>


salloc  -A sfaf -p serial

sinteractive

watch squeue -u 

scontrol show job 

job dependency -d