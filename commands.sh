ssh -X -C zhaoliu@owens.osc.edu

salloc -N <number_nodes> -A PAS2661 -pserial time=<time_in_minutes>

salloc -N <number_nodes> -A PAS2661 -p gpudebug --gpus-per-node=1 --time=<time_in_minutes>

salloc -N 1 -A PAS2661 -p gpuserial --gpus-per-node=1 --time=20


To use the bundled libc++ please add the following LDFLAGS:
  LDFLAGS="-L/usr/local/opt/llvm/lib/c++ -Wl,-rpath,/usr/local/opt/llvm/lib/c++"

llvm is keg-only, which means it was not symlinked into /usr/local,
because macOS already provides this software and installing another version in
parallel can cause all kinds of trouble.

If you need to have llvm first in your PATH, run:
  echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> /Users/liuzhao/.bash_profile

For compilers to find llvm you may need to set:
  export LDFLAGS="-L/usr/local/opt/llvm/lib"
  export CPPFLAGS="-I/usr/local/opt/llvm/include"

  export LDFLAGS="-L/usr/local/opt/libomp/lib"
  export CPPFLAGS="-I/usr/local/opt/libomp/include"


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