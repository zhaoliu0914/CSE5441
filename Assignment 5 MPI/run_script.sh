#!/bin/bash

#If you want to see what the script is doing, uncomment the following line
#set -x

module load mvapich2/2.3.3

subdir="output"
#Cleanup previous directory for submission
rm -rf $subdir
#Create a directory for submission
mkdir -p $subdir

success=1

#DO NOT CHANGE THIS
ppn=28
num_nodes=5

for program in prod_cons_mpi_hybrid
do
    term_output=$subdir/$program-terminal-output

    #Build program
    echo "Build $program" | tee -a $program-terminal-output

    #Compile the code with only mutex
    mpicc -fopenmp $program.c &> $program-compilation-output

    #Check for successful completion of the program
    if [ "$?" != 0 ]
    then
        echo "Error: There are compilation errors with $program.c" | tee -a $subdir/$program-terminal-output
        cat $program-compilation-output | tee -a $subdir/$program-terminal-output
        echo "Error: Please fix the errors and retry." | tee -a $subdir/$program-terminal-output
        exit
    else
        echo "Success: $program.c compiled fine" | tee -a $subdir/$program-terminal-output
    fi
    echo "=====================================" | tee -a $subdir/$program-terminal-output

    #Run the program with shortlist and longlist
    for input in shortlist longlist
    do
        echo "Running $program with $input" | tee -a $subdir/$program-terminal-output
        echo "=====================================" | tee -a $subdir/$program-terminal-output
        #Run the program with different number of producers
        for producers in 1 2 4 6 8
        do
            #Run the program with different number of consumers
            for consumers in 1 2 4 8
            do
                #DO NOT CHANGE THIS
                nthreads=`echo "$producers + $consumers" | bc -l`
                export MV2_THREADS_PER_PROCESS=$nthreads

                outfile="$subdir/$program-$input-output-$producers-$consumers.txt"

                #Run the program
                (time srun -N $num_nodes -n $num_nodes --time=50 ./a.out $producers $consumers < $input) &> $outfile
                if [ "$?" != 0 ]
                then
                    echo "Error: There are runtime errors with $program.c" | tee -a $subdir/$program-terminal-output
                fi

                validate_output=`python validate.py $PWD $program $input $producers $consumers`
                echo "$validate_output"
                if [[ "$(echo $validate_output | grep "Failure" | wc -l)" != "0" ]]; then
                    success=0
                fi
            done
        done
    done
done

if [[ "$success" == "1" ]]; then
    submission_dir="$PWD/cse5441-mpi-lab"
    echo "Writing submission files to $submission_dir"
    mkdir $submission_dir 2> /dev/null
    for program in prod_cons_mpi_hybrid
    do
        cp $program.c $submission_dir/
    done
fi

#Command to submit the assignment
#/fs/ess/PAS2661/CSE5441_SP24/submit
