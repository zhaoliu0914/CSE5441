#!/bin/bash

#If you want to see what the script is doing, uncomment the following line
#set -x

#DO NOT CHANGE THIS: START
module load cuda/11.6.1
module list &> _out
module_found=`cat _out | grep -i cuda | wc -l | awk '{ print $1 }'`
#Cleanup
rm -f _out
if [ "$module_found" != 1 ]
then
    echo "CUDA module not found. Are you sure you are on a GPU node?"
    echo "Use the following command to allocate a GPU node"
    echo "salloc -N <number_nodes> -A PAS2661 -p gpudebug --gpus-per-node=1 --time=<time_in_minutes>"
    exit
fi

nvidia-smi &> /dev/null
if [ "$?" != "0" ]
then
    echo "No GPUs found. Are you sure you are on a GPU node?"
    echo "Use the following command to allocate a GPU node"
    echo "salloc -N <number_nodes> -A PAS2661 -p gpudebug --gpus-per-node=1 --time=<time_in_minutes>"
    exit
fi
#DO NOT CHANGE THIS: END

subdir="cse5441-cuda-lab"

rm -rf output
mkdir -p output 2> /dev/null

success=1

for program in matrix_mul matrix_mul_shared_dynamic
do
    #Cleanup
    rm -f output/$program-terminal-output
    #Copy program to submission directory
    cp $program.cu output/
    #Build program
    echo "Build $program" | tee -a output/$program-terminal-output
    #Compile the code with only mutex
    nvcc $program.cu &> output/$program-compilation-output

    #Check for successful completion of the program
    if [ "$?" != 0 ]
    then
        echo "Error: There are compilation errors with $program.cu" | tee -a output/$program-terminal-output
        cat output/$program-compilation-output | tee -a output/$program-terminal-output
        echo "Error: Please fix the errors and retry." | tee -a output/$program-terminal-output
        exit
    else
        echo "Success: $program.cu compiled fine" | tee -a output/$program-terminal-output
    fi
    echo "=====================================" | tee -a output/$program-terminal-output

    for block_size in 4 8 16 32
    do
        outfile="output/$program-$block_size-block-output"
        (time ./a.out $block_size) &> $outfile
        if [ "$?" != 0 ]
        then
            echo "Error: There are runtime errors with $program.cu" | tee -a output/$program-terminal-output
            success=0
        else
            #Find the amount of time the program took
            duration=`grep "real" $outfile | awk '{ print $2 }'`
            compute=`grep "Compute took" $outfile | awk '{ print $4 }'`
            echo "Success: $program.cu with block size $block_size ran fine in $duration compute took $compute seconds" | tee -a output/$program-terminal-output
        fi
    done
done

if [[ "$success" == "1" ]]; then
    submission_dir="$PWD/cse5441-cuda-lab"
    echo "Writing submission files to $submission_dir"
    rm -rf $submission_dir 2> /dev/null
    mkdir $submission_dir 2> /dev/null
    for program in matrix_mul_shared_dynamic
    do
        cp $program.cu $submission_dir/
    done
fi

#Command to submit the assignment
#/fs/ess/PAS2661/CSE-5441-SP24/submit
