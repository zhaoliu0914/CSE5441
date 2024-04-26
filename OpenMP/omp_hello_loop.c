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
    int i;
    int j;
    int n = 10;
    int t = 0;

    // set 2 thread to be run
    omp_set_num_threads(2);

#pragma omp parallel private(j)
    {
#pragma omp for
        for (i = 0; i < n; i++) {
            printf("a\n");
        }

        for (j = 0; j < n; j++) {
            printf("b\n");
        }
    }
}
