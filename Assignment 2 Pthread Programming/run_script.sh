#!/bin/bash

module load gcc-compatibility/8.4.0

#If you want to see what the script is doing, uncomment the following line
#set -x

#Create a directory for output
mkdir -p output 2> /dev/null

success=1

for program in prod_consumer_mutex prod_consumer_condvar
do
    echo "Build $program"
    #Compile the code with only mutex
    gcc -pthread $program.c &> _comp_output

    #Check for successful completion of the program
    if [ "$?" != 0 ]
    then
        echo "Error: There are compilation errors with $program.c"
        cat _comp_output
        rm -f _comp_output
        echo "Error: Please fix the errors and retry."
        exit
    else
        echo "Success: $program.c compiled fine"
        rm -f _comp_output
    fi
    echo "====================================="

    #Run the program with shortlist and longlist
    for input in shortlist longlist
    do
        echo "Running $program with $input"
        echo "====================================="
        #Run the program with different number of consumers
        for consumers in 2 4 6 8
        do
            #Run the program
            (time ./a.out $consumers < $input) &> output/$program-$input-output-$consumers.txt
            validate_output=`python3 validate.py "$PWD" $program $input $consumers`
            echo $validate_output
            if [[ "$(echo $validate_output | grep "Failure" | wc -l)" != "0" ]]; then
                success=0
            fi
        done
        echo "====================================="
    done
done

if [[ "$success" == "1" ]]; then
    submission_dir="$PWD/cse5441-pthreads-lab"
    echo "Writing submission files to $submission_dir"
    mkdir cse5441-pthreads-lab 2> /dev/null
    for program in prod_consumer_mutex prod_consumer_condvar
    do
        cp $program.c $submission_dir/
    done
fi
