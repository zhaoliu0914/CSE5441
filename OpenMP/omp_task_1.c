#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdbool.h>

int main(int argc, char *argv[]) {

    omp_set_num_threads(2);

#pragma omp parallel
    {
#pragma omp task
        printf("A \n");

#pragma omp task
        printf("B \n");
    }

}