#!/bin/bash

module load gcc-compatibility/8.4.0

#If you want to see what the script is doing, uncomment the following line
#set -x

#Create a directory for output
mkdir -p output 2> /dev/null

success=1

for program in prod_consumer_omp
do
    echo "Build $program"
    #Compile the code with only mutex
    gcc-13 -fopenmp $program.c &> _comp_output

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
    for input in shortlist # longlist
    do
        echo "Running $program with $input"
        echo "====================================="
        #Run the program with different number of producers
        for producers in 1 2 4 8 16
        do
            #Run the program with different number of consumers
            for consumers in 1 2 4 8 16
            do
                #Run the program
                (time ./a.out $producers $consumers < $input) &> output/$program-$input-output-$producers-$consumers.txt
                validate_output=`python validate.py "$PWD" $program $input $producers $consumers`
                echo "$validate_output"
                if [[ "$(echo $validate_output | grep "Failure" | wc -l)" != "0" ]]; then
                    success=0
                fi
            done
        done
        echo "====================================="
    done
done

if [[ "$success" == "1" ]]; then
    submission_dir="$PWD/cse5441-omp-lab"
    echo "Writing submission files to $submission_dir"
    mkdir cse5441-omp-lab 2> /dev/null
    for program in prod_consumer_omp
    do
        cp $program.c $submission_dir/
    done
fi
