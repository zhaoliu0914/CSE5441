#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int a = 0, b = 0, i = 0, tid = 0;
float x = 0.0;

#pragma omp threadprivate(a, x)

void main() {

    printf("1st Parallel Region:\n");
#pragma omp parallel private(b, tid)
    {
        tid = omp_get_thread_num();
        a = tid;
        b = tid;
        x = 1.1 * tid + 1.0;
        printf("Thread %d:   a,b,x= %d %d %f\n", tid, a, b, x);
    }  /* end of parallel section */

    printf("************************************\n");
    printf("Master thread doing serial work here\n");
    printf("************************************\n");

    printf("2nd Parallel Region:\n");
#pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        printf("Thread %d:   a,b,x= %d %d %f\n", tid, a, b, x);
    }  /* end of parallel section */
}
