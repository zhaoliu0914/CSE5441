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

int main (int argc, char *argv[])
{
    int i = 0;
    int n = 0;

    /* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel for lastprivate(n)
    for (i = 0; i < 10; ++i) {
        n = i;
    }  /* All threads join master thread and disband */

    printf("Final: n = %d\n", n);
}
