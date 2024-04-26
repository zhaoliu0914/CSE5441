/******************************************************************************
* FILE: omp_hello_lastprivate.c
* DESCRIPTION:
*   OpenMP Example - Hello World - C/C++ Version
*   In this simple example, the master thread forks a parallel region.
*   All threads in the team obtain their unique thread number and print it.
*   The master thread only prints the total number of threads.  Two OpenMP
*   library routines are used to obtain the number of threads and each
*   thread's number.
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {

    int sum = 0;

#pragma omp parallel for reduction(+:sum)
    for (int k = 0; k < 10; k++) {
        sum += k;

        printf("k = %d, sum = %d\n", k, sum);
    }

    printf("Final: sum = %d\n", sum);
}
